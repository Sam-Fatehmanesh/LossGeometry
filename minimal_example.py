# !pip install scikit-rmt

import os
import math
import random
import pickle
import tiktoken
from datetime import datetime
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution, TracyWidomDistribution
import requests

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_TYPE = 'nanogpt'  # Choose 'mlp', 'vit', or 'nanogpt'
NUM_RUNS            = 5
NUM_EPOCHS          = 10
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-2
MOMENTUM            = 0.9
LOG_EVERY_N_BATCHES = 200

INPUT_SIZE       = 28 * 28
HIDDEN_SIZE      = 1024
OUTPUT_SIZE      = 10
NUM_HIDDEN_LAYERS = 2

DATA_ROOT = './data'

# ViT configuration parameters
VIT_IMAGE_SIZE = 28
VIT_PATCH_SIZE = 7
VIT_EMBED_DIM = 64
VIT_DEPTH = 2
VIT_NUM_HEADS = 4
VIT_MLP_RATIO = 2.0
VIT_INPUT_CHANNELS = 1
VIT_INIT_FC = True

# nanoGPT configuration parameters
NANOGPT_VOCAB_SIZE = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
NANOGPT_BLOCK_SIZE = 256
NANOGPT_N_LAYER = 4
NANOGPT_N_HEAD = 4
NANOGPT_N_EMBD = 256
NANOGPT_DROPOUT = 0.0
NANOGPT_BIAS = False
NANOGPT_LEARNING_RATE = 5e-4
NANOGPT_WEIGHT_DECAY = 0.1
NANOGPT_BETA1 = 0.9
NANOGPT_MAX_ITERS = 2000
NANOGPT_OPTIMIZER = 'sgd'
NANOGPT_WARMUP_ITERS = 200
NANOGPT_LR_DECAY_ITERS = 2000
NANOGPT_MIN_LR = 5e-5
NANOGPT_DECAY_LR = True
NANOGPT_GRAD_CLIP = 1.0

# ------------------------------------------------------------------------------
# TEXT DATA PREPARATION
# ------------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")

def download_text_sample(data_dir):
    """Download a sample text dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Use a tiny Shakespeare dataset instead - it's reliable and small
    sample_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_file = os.path.join(data_dir, "shakespeare.txt")
    
    if not os.path.exists(output_file):
        print(f"Downloading Shakespeare sample to {output_file}...")
        with requests.get(sample_url, stream=True) as r:
            r.raise_for_status()  # Ensure we catch HTTP errors
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(output_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        print(f"Download complete: {output_file}")
    else:
        print(f"File already exists: {output_file}")
    
    return output_file

def prepare_text_data(data_dir, val_split=0.1, seed=42):
    """Prepare text dataset for nanoGPT by tokenizing and converting to .bin files"""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the text sample
    text_file = download_text_sample(data_dir)
    
    # Verify the file exists and has content
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    if os.path.getsize(text_file) == 0:
        raise ValueError(f"Downloaded file is empty: {text_file}")
    
    # Read the text file and split into documents
    print("Processing text data...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # For Shakespeare, split by newlines to get reasonable chunks
    raw_documents = [doc.strip() for doc in text.split('\n') if doc.strip()]
    
    # Combine short lines to form more substantial documents
    documents = []
    current_doc = []
    for line in raw_documents:
        current_doc.append(line)
        # If we've accumulated about 100 tokens or hit a scene break, create a document
        if len(' '.join(current_doc)) > 400 or line.strip() == '':
            if current_doc:  # Ensure we don't add empty documents
                documents.append(' '.join(current_doc))
                current_doc = []
    
    # Add any remaining lines
    if current_doc:
        documents.append(' '.join(current_doc))
    
    print(f"Split text into {len(documents)} documents")
    
    # Shuffle the documents
    random.shuffle(documents)
    
    # Split into train/val
    n_val = max(1, int(len(documents) * val_split))
    train_documents = documents[:-n_val]
    val_documents = documents[-n_val:]
    
    # Tokenize
    print(f"Tokenizing {len(train_documents)} train documents...")
    train_tokens = []
    for doc in tqdm(train_documents):
        train_tokens.extend(enc.encode(doc))
        # Add a separator token (newline)
        train_tokens.append(enc.eot_token)  # EOT token
    
    print(f"Tokenizing {len(val_documents)} validation documents...")
    val_tokens = []
    for doc in tqdm(val_documents):
        val_tokens.extend(enc.encode(doc))
        # Add a separator token (newline)
        val_tokens.append(enc.eot_token)  # EOT token
    
    # Convert to uint16 arrays and save
    print(f"Train set: {len(train_tokens)} tokens")
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    save_path = os.path.join(data_dir, 'train.bin')
    train_tokens.tofile(save_path)
    
    print(f"Validation set: {len(val_tokens)} tokens")
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    save_path = os.path.join(data_dir, 'val.bin')
    val_tokens.tofile(save_path)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens)
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        import pickle
        pickle.dump(meta, f)
    
    print("Done!")

# ------------------------------------------------------------------------------
# MODEL DEFINITION
# ------------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc_layers.append(nn.Linear(hidden_size, output_size))
        self.relu = nn.ReLU()
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
        for layer in self.fc_layers:
            fan_in = layer.weight.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers[:-1]:
            x = self.relu(layer(x))
        return self.fc_layers[-1](x)

    def get_target_layers(self):
        return [f"fc_layers.{i}.weight" for i in range(len(self.fc_layers))]

    def get_parameter(self, name):
        parts = name.split('.')
        obj = self
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class CustomViT(nn.Module):
    """
    Very small Vision Transformer (ViT) wrapper for LossGeometry.
    Implements patch embedding, transformer encoder, and classification head.
    """
    def __init__(self,
                 num_classes=10,
                 image_size=28,
                 patch_size=7,
                 embed_dim=64,
                 depth=2,
                 num_heads=4,
                 mlp_ratio=2.0,
                 input_channels=1,
                 gaussian_init_fc=False):
        super().__init__()
        # Expose attributes for compatibility
        self.input_size = input_channels
        self.hidden_size = embed_dim
        self.output_size = num_classes
        self.num_hidden_layers = depth

        # Patch embedding module
        self.patch_embed = PatchEmbed(image_size, patch_size, input_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=0.0)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embed)
        if gaussian_init_fc:
            fan_in = self.head.weight.data.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(self.head.weight.data, mean=0.0, std=std)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias.data)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x).transpose(0, 1)
        x = self.transformer(x)
        x = x[0]
        x = self.head(x)
        return x

    def get_target_layers(self):
        layers = ['head.weight']
        for i in range(len(self.transformer.layers)):
            layers.append(f'transformer.layers.{i}.linear1.weight')
            layers.append(f'transformer.layers.{i}.linear2.weight')
        return layers

    def get_parameter(self, layer_name):
        parts = layer_name.split('.')
        obj = self
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPTSpectral(nn.Module):
    """GPT model extended with methods to expose layers for spectral analysis"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Add these attributes to make compatible with save_analysis_data
        self.input_size = config.vocab_size
        self.hidden_size = config.n_embd
        self.output_size = config.vocab_size
        self.num_hidden_layers = config.n_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying - note this needs to be done after applying _init_weights or it will be overwritten
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Now set up weight tying
        self.transformer.wte.weight = self.lm_head.weight # weight tying
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
        # Create a mapping of layer names to parameter names for spectral analysis
        self._target_layers = []
        self._build_target_layers()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_target_layers(self):
        """Build a list of target layers for spectral analysis"""
        # Add attention weight matrices
        for i in range(self.config.n_layer):
            self._target_layers.append(f"transformer.h.{i}.attn.c_attn.weight")
            self._target_layers.append(f"transformer.h.{i}.attn.c_proj.weight")
            
        # Add MLP weight matrices
        for i in range(self.config.n_layer):
            self._target_layers.append(f"transformer.h.{i}.mlp.c_fc.weight")
            self._target_layers.append(f"transformer.h.{i}.mlp.c_proj.weight")
            
        # Add embedding - but note that embedding weight is shared with lm_head
        self._target_layers.append("transformer.wte.weight")
        
        # We don't need to add lm_head.weight since it shares parameters with transformer.wte.weight
        # due to weight tying, and will just cause confusion during analysis

    def get_target_layers(self):
        """Return the list of target layers for spectral analysis"""
        return self._target_layers
        
    def get_parameter(self, param_name):
        """Get a parameter by name"""
        for name, param in self.named_parameters():
            if name == param_name:
                return param
        raise AttributeError(f"Parameter '{param_name}' not found in model")

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # The number of transformer layers
        L = self.config.n_layer
        # The hidden dimension
        H = self.config.n_embd
        # The number of heads
        Q = self.config.n_head
        # The batch size
        B = 1  # Assuming batch size of 1 for simplicity
        # The sequence length
        T = self.config.block_size
        
        # estimate the number of flops we do per iteration.
        # see the GPT-3 paper Appendix C as ref: https://arxiv.org/abs/2005.14165, GPT-3 training_flops
        flops_per_token = 6 * L * H**2 * (1 + T/H + (Q*T)/(4*H) + Q/4)
        flops_per_fwdbwd = flops_per_token * B * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type='adamw'):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer_type_lower = optimizer_type.lower()
        
        if optimizer_type_lower == 'sgd':
            # Use SGD optimizer with momentum
            print(f"using SGD optimizer with momentum={betas[0]}")
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0])
        elif optimizer_type_lower == 'sgd_no_momentum':
            # Use SGD optimizer without momentum
            print(f"using SGD optimizer without momentum")
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.0)
        else:
            # Use AdamW optimizer (default)
            use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
        
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Initialize a pretrained GPT model by copying over the weights from a GPT-2 HuggingFace model"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from model import GPT
        
        # Start with a standard GPT model
        gpt = GPT.from_pretrained(model_type, override_args)
        
        # Create our spectral-enabled model with the same config
        config = gpt.config
        model = cls(config)
        
        # Copy over the weights
        model.load_state_dict(gpt.state_dict())
        
        return model
        
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size] 

if MODEL_TYPE == 'mlp':
    _reference_model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
elif MODEL_TYPE == 'vit':
    _reference_model = CustomViT(
        num_classes=OUTPUT_SIZE,
        image_size=VIT_IMAGE_SIZE,
        patch_size=VIT_PATCH_SIZE,
        embed_dim=VIT_EMBED_DIM,
        depth=VIT_DEPTH,
        num_heads=VIT_NUM_HEADS,
        mlp_ratio=VIT_MLP_RATIO,
        input_channels=VIT_INPUT_CHANNELS,
        gaussian_init_fc=VIT_INIT_FC
    )
elif MODEL_TYPE == 'nanogpt':
    _nanogpt_config = GPTConfig(
        block_size=NANOGPT_BLOCK_SIZE,
        vocab_size=NANOGPT_VOCAB_SIZE,
        n_layer=NANOGPT_N_LAYER,
        n_head=NANOGPT_N_HEAD,
        n_embd=NANOGPT_N_EMBD,
        dropout=NANOGPT_DROPOUT,
        bias=NANOGPT_BIAS
    )
    _reference_model = GPTSpectral(_nanogpt_config)
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
LAYER_NAMES = _reference_model.get_target_layers()
LAYER_SHAPES = [tuple(_reference_model.get_parameter(name).shape) for name in LAYER_NAMES]

# ------------------------------------------------------------------------------
# TEXT DATA LOADING FOR NANOGPT (adapted from train_spectral.py)
# ------------------------------------------------------------------------------
def get_batch_text(split, block_size, batch_size, device):
    """Load training and validation data for nanoGPT"""
    # We recreate np.memmap every batch to avoid a memory leak
    data_dir = DATA_ROOT
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
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
def estimate_loss_nanogpt(model, device, eval_iters=50):
    """Estimate loss for nanoGPT model on train/val splits"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_text(split, NANOGPT_BLOCK_SIZE, BATCH_SIZE, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ------------------------------------------------------------------------------
# TRAIN + AGGREGATE SINGULAR VALUES
# ------------------------------------------------------------------------------
def train_and_aggregate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data loaders based on model type
    if MODEL_TYPE == 'nanogpt':
        # Ensure text data is prepared
        prepare_text_data(DATA_ROOT)
        print("Text data prepared for nanoGPT training")
    else:
        # MNIST loader (exact same normalize as main.py)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

    aggregated_sv   = None
    reference_batches = None

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
        if MODEL_TYPE == 'mlp':
            model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS).to(device)
            # Use standard optimizer settings
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
            criterion = nn.CrossEntropyLoss()
        elif MODEL_TYPE == 'vit':
            model = CustomViT(
                num_classes=OUTPUT_SIZE,
                image_size=VIT_IMAGE_SIZE,
                patch_size=VIT_PATCH_SIZE,
                embed_dim=VIT_EMBED_DIM,
                depth=VIT_DEPTH,
                num_heads=VIT_NUM_HEADS,
                mlp_ratio=VIT_MLP_RATIO,
                input_channels=VIT_INPUT_CHANNELS,
                gaussian_init_fc=VIT_INIT_FC
            ).to(device)
            # Use standard optimizer settings
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
            criterion = nn.CrossEntropyLoss()
        elif MODEL_TYPE == 'nanogpt':
            # Create nanoGPT model
            nanogpt_config = GPTConfig(
                block_size=NANOGPT_BLOCK_SIZE,
                vocab_size=NANOGPT_VOCAB_SIZE,
                n_layer=NANOGPT_N_LAYER,
                n_head=NANOGPT_N_HEAD,
                n_embd=NANOGPT_N_EMBD,
                dropout=NANOGPT_DROPOUT,
                bias=NANOGPT_BIAS
            )
            model = GPTSpectral(nanogpt_config).to(device)
            # Use nanoGPT-specific optimizer settings (adapted from train_spectral.py)
            optimizer = model.configure_optimizers(
                weight_decay=NANOGPT_WEIGHT_DECAY, 
                learning_rate=NANOGPT_LEARNING_RATE, 
                betas=(NANOGPT_BETA1, 0.95),
                device_type=device.type,
                optimizer_type=NANOGPT_OPTIMIZER
            )
            criterion = None  # nanoGPT computes loss internally
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

        # Set up data loader based on model type
        if MODEL_TYPE == 'nanogpt':
            # nanoGPT doesn't use traditional DataLoader, we'll manually iterate
            loader = None
            total_batches = NANOGPT_MAX_ITERS
        else:
            loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
            total_batches = NUM_EPOCHS * len(loader)

        # collect per-layer snapshots this run
        layer_names = model.get_target_layers()
        sv_snapshots = {name: [] for name in layer_names}
        batch_list   = []
        batch_idx    = 0

        if MODEL_TYPE == 'nanogpt':
            # nanoGPT training loop (adapted from train_spectral.py)
            pbar = tqdm(range(NANOGPT_MAX_ITERS), desc=f"Training GPT Run {run+1}")
            for iter_num in pbar:
                # Get batch
                X, Y = get_batch_text('train', NANOGPT_BLOCK_SIZE, BATCH_SIZE, device)
                
                # Determine and set the learning rate for this iteration
                if NANOGPT_DECAY_LR:
                    lr = get_lr(iter_num, NANOGPT_LEARNING_RATE, NANOGPT_WARMUP_ITERS, 
                               NANOGPT_LR_DECAY_ITERS, NANOGPT_MIN_LR)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                # Forward pass
                optimizer.zero_grad()
                logits, loss = model(X, Y)
                loss.backward()
                
                # Gradient clipping
                if NANOGPT_GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), NANOGPT_GRAD_CLIP)
                
                # snapshot
                if batch_idx % LOG_EVERY_N_BATCHES == 0:
                    for name in layer_names:
                        with torch.no_grad():
                            # move weight matrix to CPU to avoid GPU SVD nan issues
                            W_cpu = model.get_parameter(name).data.cpu()
                            # center the matrix
                            W_center = W_cpu - W_cpu.mean()
                            sigma = W_center.std()
                            if sigma > 1e-15:
                                # normalize by sigma * sqrt(max_dim) to match MP theory
                                max_dim = float(max(W_center.shape))
                                W_norm = W_center / (sigma * math.sqrt(max_dim))
                                sv = torch.linalg.svdvals(W_norm).numpy()
                                # filter out any non-finite values
                                sv = sv[np.isfinite(sv)]
                            else:
                                sv = np.array([], dtype=float)
                        sv_snapshots[name].append(sv)
                    batch_list.append(batch_idx)

                optimizer.step()
                batch_idx += 1
                
                # Periodic evaluation (every 100 iterations)
                if iter_num % 100 == 0 and iter_num > 0:
                    losses = estimate_loss_nanogpt(model, device, eval_iters=20)
                    print(f"\nRun {run+1}, iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Update progress bar with loss and learning rate
                if NANOGPT_DECAY_LR:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
                else:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        else:
            # Original training loop for MLP/ViT
            for epoch in range(NUM_EPOCHS):
                pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
                for imgs, labels in pbar:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                    loss.backward()

                    # snapshot
                    if batch_idx % LOG_EVERY_N_BATCHES == 0:
                        for name in layer_names:
                            with torch.no_grad():
                                # move weight matrix to CPU to avoid GPU SVD nan issues
                                W_cpu = model.get_parameter(name).data.cpu()
                                # center the matrix
                                W_center = W_cpu - W_cpu.mean()
                                sigma = W_center.std()
                                if sigma > 1e-15:
                                    # normalize by sigma * sqrt(max_dim) to match MP theory
                                    max_dim = float(max(W_center.shape))
                                    W_norm = W_center / (sigma * math.sqrt(max_dim))
                                    sv = torch.linalg.svdvals(W_norm).numpy()
                                    # filter out any non-finite values
                                    sv = sv[np.isfinite(sv)]
                                else:
                                    sv = np.array([], dtype=float)
                            sv_snapshots[name].append(sv)
                        batch_list.append(batch_idx)

                    optimizer.step()
                    batch_idx += 1

        # first run initializes aggregator
        if aggregated_sv is None:
            aggregated_sv    = {name: [sv_snapshots[name]] for name in sv_snapshots}
            reference_batches = batch_list
        else:
            for name in sv_snapshots:
                aggregated_sv[name].append(sv_snapshots[name])

    # concatenate across runs
    final_sv = {}
    for name, runs_list in aggregated_sv.items():
        num_snaps = len(runs_list[0])
        combined = []
        for snap in range(num_snaps):
            arrays = [ runs_list[r][snap] for r in range(NUM_RUNS) ]
            combined.append(np.concatenate(arrays))
        final_sv[name] = combined

    # Calculate layer shapes from the actual model used
    layer_shapes = [tuple(model.get_parameter(name).shape) for name in layer_names]
    
    return final_sv, reference_batches, layer_names, layer_shapes


# ------------------------------------------------------------------------------
# PLOTTING 20 EQUALLY-SPACED HISTOGRAMS PER LAYER
# ------------------------------------------------------------------------------
def plot_singular_evolution(final_sv, batches, layer_names, layer_shapes):
    for name in layer_names:
        # Advanced singular-value plotting with MP & TW overlays
        results_data = final_sv[name]
        num_plots    = min(20, len(results_data))
        indices      = np.linspace(0, len(results_data)-1, num_plots, dtype=int)
        # Determine common range robustly, ignoring NaNs
        all_vals = np.concatenate([results_data[i] for i in indices if len(results_data[i]) > 0])
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            max_val = np.max(finite_vals)
            max_plot = max(max_val * 1.1, 3.0)
        else:
            max_plot = 3.0
        common_range = (0, max_plot)
        ncols = 5
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
        axes = axes.flatten()
        # Get layer shape
        idx = layer_names.index(name)
        m, n = layer_shapes[idx]
        ratio = min(m,n)/max(m,n)
        is_square = (m==n)
        for pi, snap in enumerate(indices):
            ax = axes[pi]
            sv_vals = results_data[snap]
            nb = max(min(len(sv_vals)//10, 100), 30)
            counts, bins, _ = ax.hist(sv_vals, bins=nb, density=False,
                                     alpha=0.7, color='royalblue',
                                     label='Histogram', range=common_range)
            w = bins[1]-bins[0]
            centers = (bins[:-1]+bins[1:])/2
            # MP overlay
            mp = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)
            x_mp = np.linspace(common_range[0], common_range[1], 500)
            mp_d = np.array([2*x*mp.pdf(x*x) if x>0 else 0 for x in x_mp])
            ax.plot(x_mp, mp_d * len(sv_vals)*w, 'r--', linewidth=1.5, label=f'MP (λ={ratio:.2f})')
            # TW overlay
            tw = TracyWidomDistribution(beta=1)
            edge = 2.0
            scale = 0.5*(np.sqrt(m)+np.sqrt(n))**(-1/3)
            x_tw = np.linspace(edge-4*scale, edge+4*scale, 300)
            args = (x_tw-edge)/scale
            pdf_tw = tw.pdf(args)/scale
            mpv = np.max(pdf_tw)
            mask = (centers>edge-scale)&(centers<edge+scale)
            peak = np.max(counts[mask]) if np.any(mask) else np.max(counts)
            ax.plot(x_tw, pdf_tw*peak/mpv, 'g-', linewidth=2.0,
                    label=f'TW (edge={edge:.2f}, scale={scale:.2e})')
            formula = (r"$A_k(\tau)\approx\frac{\sigma_{(k)}-2\sqrt{n}}{2^{1/3}n^{1/6}}$"
                       if is_square else
                       r"$A_k(\tau)\approx\frac{\sigma_{(k)}-(\sqrt{m}+\sqrt{n})}{(\sqrt{m}+\sqrt{n})^{1/2}(m^{-1/2}+n^{-1/2})^{1/3}}$")
            ax.text(0.5, 0.93, formula, transform=ax.transAxes,
                    horizontalalignment='center', fontsize=10,
                    bbox=dict(facecolor='white',alpha=0.7))
            ax.set_title(f"Batch {batches[snap]}")
            ax.set_xlim(common_range)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Singular Value (σ)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
        for ax in axes[num_plots:]: ax.axis('off')
        fig.suptitle(f"Singular-Value Evolution for {name} (runs={NUM_RUNS})", fontsize=14)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()

# ------------------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------------------
print(f"Running spectral analysis with model type: {MODEL_TYPE}")
start = datetime.now()
final_sv, batches, layers, layer_shapes = train_and_aggregate()
print(f"\nTotal time: {datetime.now() - start}\n")
plot_singular_evolution(final_sv, batches, layers, layer_shapes)
