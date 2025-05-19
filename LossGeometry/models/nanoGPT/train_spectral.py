"""
Train nanoGPT with spectral analysis on eigenvalues and singular values.

This script is an extension of the original train.py, adding spectral analysis
of the model's weight matrices during training.

Example:
    $ python train_spectral.py config/train_pile_spectral.py
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import pathlib
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_spectral import GPTSpectral, GPTConfig

# Add parent directory to path for spectral_analysis import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from LossGeometry.analysis.spectral_analysis import SpectralAnalyzer
from LossGeometry.utils.io_utils import save_analysis_data
from datetime import datetime

# -----------------------------------------------------------------------------
# Default config values - will be overridden by config file
# I/O
out_dir = 'out-spectral'
eval_interval = 250
save_interval = 500
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'pile_spectral'
wandb_run_name = 'nanogpt-pile-spectral'
# data
dataset = 'pile'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 1024
# model
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 3e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Disable compilation by default for spectral analysis
# spectral analysis specific
enable_spectral_analysis = True
spectral_analysis_interval = 100  # Every 100 iterations

# Load config overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# Various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # Also create a directory for spectral analysis results
    spectral_dir = os.path.join(out_dir, 'spectral')
    os.makedirs(spectral_dir, exist_ok=True)
    
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader: Load training and validation data
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPTSpectral(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPTSpectral(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPTSpectral.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# Initialize the spectral analyzer if enabled
if enable_spectral_analysis and master_process:
    spectral_analyzer = SpectralAnalyzer(
        analyze_W=True, 
        analyze_delta_W=True,
        analyze_spectral_density=True,
        analyze_level_spacing=False,
        analyze_singular_values=True
    )
    # Get the list of target layers from the model
    target_layers = model.get_target_layers()
    spectral_analyzer.initialize_layer_stats(target_layers)
else:
    spectral_analyzer = None

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
train_losses = []
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
batch_spectral_triggered = False

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,  # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # Perform spectral analysis at specific intervals
    if enable_spectral_analysis and master_process and iter_num % spectral_analysis_interval == 0:
        print(f"\n--- Performing spectral analysis at iteration {iter_num} ---")
        recorded = spectral_analyzer.analyze_batch(raw_model, optimizer, iter_num)
        if recorded:
            print(f"Recorded spectral analysis data for iteration {iter_num}")
            # Track the loss value for correlation with spectral properties
            if len(train_losses) > 0:
                avg_loss = sum(train_losses) / len(train_losses)
                spectral_analyzer.track_loss(avg_loss, iter_num)
                train_losses = []  # Reset for next interval
            
            # Save spectral analysis results periodically
            if iter_num % save_interval == 0:
                print(f"Saving spectral analysis data to {spectral_dir}")
                stats = spectral_analyzer.get_stats()
                torch.save(stats, os.path.join(spectral_dir, f'spectral_stats_{iter_num}.pt'))
                
                # Also save in HDF5 format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                h5_path = save_analysis_data(spectral_analyzer, raw_model, out_dir, timestamp)
                print(f"Saved spectral analysis data in HDF5 format to: {h5_path}")
        batch_spectral_triggered = True

    if iter_num == 0 and enable_spectral_analysis and master_process:
        # Analyze initial weights
        print(f"\n--- Performing initial spectral analysis ---")
        spectral_analyzer.analyze_batch(raw_model, optimizer, 0)
        print("Completed initial spectral analysis")

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            # Track losses for correlation with spectral properties
            if enable_spectral_analysis and master_process:
                train_losses.append(loss.item())
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above
        lossf = loss.item()
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    
    # Check if we've reached the maximum number of iterations
    if iter_num >= max_iters:
        break

# Save the final spectral analysis results
if enable_spectral_analysis and master_process:
    print(f"Saving final spectral analysis data to {spectral_dir}")
    stats = spectral_analyzer.get_stats()
    torch.save(stats, os.path.join(spectral_dir, 'spectral_stats_final.pt'))
    
    # Also save in HDF5 format like main.py does
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp += "_final"  # Add final marker
    h5_path = save_analysis_data(spectral_analyzer, raw_model, out_dir, timestamp)
    print(f"Saved final spectral analysis data in HDF5 format to: {h5_path}")

# Save the final model
if master_process:
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt_final.pt'))

if ddp:
    destroy_process_group() 