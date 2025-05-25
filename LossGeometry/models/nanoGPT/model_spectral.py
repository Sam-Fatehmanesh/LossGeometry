"""
Extended version of nanoGPT model with additional functionality for spectral analysis.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig

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