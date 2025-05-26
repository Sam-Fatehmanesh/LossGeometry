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
from tqdm import tqdm

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
learning_rate = 0.01
max_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
optimizer = 'sgd'  # SGD optimizer with momentum
# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 2000
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
# multiple runs
num_runs = 1  # Default to 1 run

# Load config overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# Data loader: Load training and validation data
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak
    data_dir = os.path.join('data', dataset)
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

def setup_training_environment():
    """Initialize the training environment - directories, DDP, etc."""
    global device_type, ctx, master_process, ddp_world_size, gradient_accumulation_steps, device, spectral_dir
    
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
        
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize torch seeds for reproducibility
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    return seed_offset, ddp

def init_new_model():
    """Initialize a fresh model instance"""
    # Create model with global configuration
    global model, optimizer_instance, scaler, raw_model
    
    # determine the vocab size we'll use for from-scratch training
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line

    # Determine vocab size from meta file or use default
    meta_vocab_size = None
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        if master_process:
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPTSpectral(gptconf)
    model.to(device)

    # Init optimizer
    optimizer_instance = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, optimizer)
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Compile if enabled
    if compile:
        model = torch.compile(model)
    
    # Wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Get raw model for analysis
    raw_model = model.module if ddp else model

    torch.manual_seed(1337 + seed_offset)
    
    if master_process:
        if init_from == 'scratch':
            print("Initializing a new model from scratch")
            print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Using device: {device}")
        if device_type == 'cuda':
            optimizer_lower = optimizer.lower()
            if optimizer_lower == 'sgd':
                print(f"using SGD optimizer with momentum: {isinstance(optimizer_instance, torch.optim.SGD)}")
            elif optimizer_lower == 'sgd_no_momentum':
                print(f"using SGD optimizer without momentum: {isinstance(optimizer_instance, torch.optim.SGD)}")
            else:
                print(f"using fused AdamW: {hasattr(torch.optim, 'fused') and isinstance(optimizer_instance, torch.optim.AdamW)}")
    
    return model_args

def train_run(run_idx):
    """Perform a single training run"""
    global iter_num, best_val_loss
    
    # Reset for this run
    iter_num = 0
    best_val_loss = 1e9
    
    # Create new model and optimizer for this run
    model_args = init_new_model()
    
    # Initialize the spectral analyzer for this run
    if enable_spectral_analysis and master_process:
        spectral_analyzer = SpectralAnalyzer(
            analyze_W=True, 
            analyze_delta_W=True,
            analyze_spectral_density=True,
            analyze_level_spacing=False,
            analyze_singular_values=True
        )
        # Get the list of target layers from the model
        target_layers = raw_model.get_target_layers()
        spectral_analyzer.initialize_layer_stats(target_layers)
    else:
        spectral_analyzer = None

    # Training stats
    train_losses = []
    batch_spectral_triggered = False
    running_mfu = -1.0
    
    # Logging setup
    if wandb_log and master_process and run_idx == 0:  # Only log first run to wandb
        import wandb
        wandb.init(project=wandb_project, name=f"{wandb_run_name}-run{run_idx}", config=config)

    # Initialize with first batch
    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    
    # Initial spectral analysis
    if iter_num == 0 and enable_spectral_analysis and master_process:
        # Analyze initial weights
        print(f"\n--- Performing initial spectral analysis for run {run_idx+1} ---")
        spectral_analyzer.analyze_batch(raw_model, optimizer_instance, 0)
        print("Completed initial spectral analysis")
    
    # Main training loop
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer_instance.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"Run {run_idx+1}/{num_runs}, step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log and run_idx == 0:  # Only log first run to wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,  # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                # Only save the final checkpoint to save disk space, not intermediate ones
                # (Code to save intermediate checkpoints is removed)
        
        # Perform spectral analysis at specific intervals
        if enable_spectral_analysis and master_process and iter_num % spectral_analysis_interval == 0:
            print(f"\n--- Performing spectral analysis for run {run_idx+1}/{num_runs} at iteration {iter_num} ---")
            recorded = spectral_analyzer.analyze_batch(raw_model, optimizer_instance, iter_num)
            if recorded:
                # Track the loss value for correlation with spectral properties
                if len(train_losses) > 0:
                    avg_loss = sum(train_losses) / len(train_losses)
                    spectral_analyzer.track_loss(avg_loss, iter_num)
                    train_losses = []  # Reset for next interval
                
                # Skip saving intermediate spectral analysis results to save disk space
            batch_spectral_triggered = True

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
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
            scaler.unscale_(optimizer_instance)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer_instance)
        scaler.update()
        
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer_instance.zero_grad(set_to_none=True)

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
                print(f"Run {run_idx+1}/{num_runs}, iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1
        
        # Check if we've reached the maximum number of iterations
        if iter_num >= max_iters:
            break

    # Save the final spectral analysis results
    if enable_spectral_analysis and master_process:
        run_spectral_dir = os.path.join(spectral_dir, f'run_{run_idx+1}')
        if not os.path.exists(run_spectral_dir):
            os.makedirs(run_spectral_dir)
        
        # Skip saving the PT file to save disk space
        # stats = spectral_analyzer.get_stats()
        # torch.save(stats, os.path.join(run_spectral_dir, 'spectral_stats_final.pt'))
        
        # Save only in HDF5 format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp += f"_run{run_idx+1}_final"
        h5_path = save_analysis_data(spectral_analyzer, raw_model, out_dir, timestamp)

    # # Save the final model
    # if master_process:
    #     checkpoint = {
    #         'model': raw_model.state_dict(),
    #         'optimizer': optimizer_instance.state_dict(),
    #         'model_args': model_args,
    #         'iter_num': iter_num,
    #         'best_val_loss': best_val_loss,
    #         'config': config,
    #         'run_idx': run_idx,
    #     }
    #     run_dir = os.path.join(out_dir, f'run_{run_idx+1}')
    #     if not os.path.exists(run_dir):
    #         os.makedirs(run_dir)
    #     torch.save(checkpoint, os.path.join(run_dir, 'ckpt_final.pt'))
    
    return spectral_analyzer

def aggregate_spectral_results(analyzers):
    """Aggregate results from multiple runs"""
    if not master_process or not enable_spectral_analysis:
        return
    
    print("\nAggregating results from all runs...")
    
    # Create a new analyzer for aggregated results
    aggregated_analyzer = SpectralAnalyzer(
        analyze_W=True,
        analyze_delta_W=True,
        analyze_spectral_density=True,
        analyze_level_spacing=False,
        analyze_singular_values=True
    )
    
    # For simplicity, use batches from first run
    first_stats = analyzers[0].stats
    target_layers = list(first_stats['results'].keys())
    aggregated_analyzer.initialize_layer_stats(target_layers)
    
    # Copy batch info from first run
    aggregated_analyzer.stats['batch'] = first_stats['batch']
    aggregated_analyzer.stats['batch_numbers'] = first_stats['batch_numbers']
    
    # Average loss values across runs
    all_losses = [analyzer.stats['loss_values'] for analyzer in analyzers]
    if all_losses and all(losses for losses in all_losses):
        avg_losses = np.mean(all_losses, axis=0)
        aggregated_analyzer.stats['loss_values'] = avg_losses.tolist()
    
    # Process each layer
    for layer_name in target_layers:
        if layer_name not in aggregated_analyzer.stats['results']:
            aggregated_analyzer.stats['results'][layer_name] = {}
        
        # Aggregate eigenvalues
        all_eigenvalues_lists = []
        for analyzer in analyzers:
            if 'eigenvalues_list' in analyzer.stats['results'].get(layer_name, {}):
                all_eigenvalues_lists.append(analyzer.stats['results'][layer_name]['eigenvalues_list'])
        
        if all_eigenvalues_lists:
            # Create an aggregated list of eigenvalues
            eigenvalues_list = []
            min_length = min(len(evs) for evs in all_eigenvalues_lists)
            
            for i in range(min_length):
                batch_eigenvalues = [evs[i] for evs in all_eigenvalues_lists]
                # Concatenate eigenvalues from all runs at this batch point
                all_eigenvalues = np.concatenate(batch_eigenvalues)
                eigenvalues_list.append(all_eigenvalues)
            
            aggregated_analyzer.stats['results'][layer_name]['eigenvalues_list'] = eigenvalues_list
            if eigenvalues_list:
                aggregated_analyzer.stats['results'][layer_name]['last_eigenvalues'] = eigenvalues_list[-1]
        
        # Aggregate singular values
        all_sv_lists = []
        for analyzer in analyzers:
            if 'singular_values_list' in analyzer.stats['results'].get(layer_name, {}):
                all_sv_lists.append(analyzer.stats['results'][layer_name]['singular_values_list'])
        
        if all_sv_lists:
            # Create an aggregated list of singular values
            sv_list = []
            min_length = min(len(svs) for svs in all_sv_lists)
            
            for i in range(min_length):
                batch_sv = [svs[i] for svs in all_sv_lists]
                # Concatenate singular values from all runs at this batch point
                all_sv = np.concatenate(batch_sv)
                sv_list.append(all_sv)
            
            aggregated_analyzer.stats['results'][layer_name]['singular_values_list'] = sv_list
            if sv_list:
                aggregated_analyzer.stats['results'][layer_name]['last_singular_values'] = sv_list[-1]
    
    # Save aggregated results
    if master_process:
        aggregated_dir = os.path.join(out_dir, 'aggregated')
        if not os.path.exists(aggregated_dir):
            os.makedirs(aggregated_dir)
        
        # Skip saving the PT file to save disk space
        # stats = aggregated_analyzer.get_stats()
        # torch.save(stats, os.path.join(aggregated_dir, 'spectral_stats_aggregated.pt'))
        
        # Save in HDF5 format only
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp += f"_aggregated_{num_runs}_runs"
        h5_path = save_analysis_data(aggregated_analyzer, raw_model, out_dir, timestamp)
        print(f"Saved aggregated spectral analysis data to: {h5_path}")
    
    return aggregated_analyzer

# Main training function to support multiple runs
def main():
    # Initialize training environment (directories, DDP setup)
    global seed_offset, ddp, master_process, spectral_dir, device_type
    seed_offset, ddp = setup_training_environment()
    
    print(f"Starting {num_runs} run(s) of nanoGPT with spectral analysis")
    
    # List to store analyzers from each run
    all_analyzers = []
    
    # Run the training loop for each run
    for run_idx in range(num_runs):
        print(f"\n======== Starting Run {run_idx+1}/{num_runs} ========")
        analyzer = train_run(run_idx)
        if analyzer is not None:
            all_analyzers.append(analyzer)
    
    # If we have multiple runs, aggregate results
    if num_runs > 1 and all_analyzers:
        aggregate_spectral_results(all_analyzers)
    
    # Clean up
    if ddp:
        destroy_process_group() 
    
    print(f"Completed {num_runs} run(s) successfully!")

if __name__ == "__main__":
    main() 