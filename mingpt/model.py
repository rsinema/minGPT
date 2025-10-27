"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# -----------------------------------------------------------------------------
# SigGLU
class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dim_out=None, bias=True):
        super().__init__()
        dim_hidden = dim_hidden or 4 * dim_in
        dim_out = dim_out or dim_in
        
        # Linear transformations for gating
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_hidden, bias=bias)
        
        # Output projection
        self.w3 = nn.Linear(dim_hidden, dim_out, bias=bias)
    
    def forward(self, x):
        # SwiGLU applies SiLU activation to one branch and gates it with the other
        hidden1 = self.w1(x)
        hidden2 = self.w2(x)
        
        # SiLU (Swish) activation: x * sigmoid(x)
        hidden1_act = hidden1 * torch.sigmoid(hidden1)
        
        # Element-wise product for gating
        hidden = hidden1_act * hidden2
        
        # Output projection
        return self.w3(hidden)

# -----------------------------------------------------------------------------
# Rotary Positional Embeddings

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, interleaved=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        
        # Generate inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len, device=None):
        # Get device from buffer if not specified
        if device is None:
            device = self.inv_freq.device
            
        # Generate position indices
        positions = torch.arange(seq_len, device=device).float()
        
        # Compute sinusoidal patterns
        freqs = torch.outer(positions, self.inv_freq)
        
        # Get sine and cosine embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb)[:, :self.dim]
        sin = torch.sin(emb)[:, :self.dim]
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, interleaved=False):
    # Apply rotary embeddings to queries and keys
    batch_size, num_heads, seq_len, head_dim = q.shape
    cos = cos.reshape(1, 1, seq_len, cos.shape[-1])  # [1, 1, seq_len, dim/2]
    sin = sin.reshape(1, 1, seq_len, sin.shape[-1])  # [1, 1, seq_len, dim/2]
    
    # Split queries and keys for rotation
    half_dim = head_dim // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    # Apply rotation using half-dim rotary embeddings
    q_rotated = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin
    ], dim=-1)
    
    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin
    ], dim=-1)
    
    return q_rotated, k_rotated

# -----------------------------------------------------------------------------
# Linear Warmup Scheduler
class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.p = True
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: use base learning rate
            return self.base_lrs
        
# -----------------------------------------------------------------------------
# Cosine Annealing Scheduler with Warmup
class CosineAnnealingWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: cosine decay from base LR to min_lr
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            # Cosine decay formula: min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
            return [base_lr * scale for base_lr in self.base_lrs]

# -----------------------------------------------------------------------------
# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x):
        # Calculate root mean square along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_normalized = x / rms
        
        # Apply scaling if using learnable parameters
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
        
        return x_normalized
    
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.rope = config.rope
        if self.rope:
            self.rotary_emb = RotaryEmbedding(dim=(config.n_embd // config.n_head) // 2)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            cos, sin = self.rotary_emb(T, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, interleaved=False)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        if config.rms_norm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)
        
        self.attn = CausalSelfAttention(config)
        if config.swiglu:
            self.mlp = nn.ModuleDict(dict(
                c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
                act     = SwiGLU(dim_in=config.n_embd * 4),
                dropout = nn.Dropout(config.resid_pdrop),
            ))
        else:
            self.mlp = nn.ModuleDict(dict(
                c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
                act     = NewGELU(),
                dropout = nn.Dropout(config.resid_pdrop),
            ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt2'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1

        C.swiglu = False  # whether to use SwiGLU activations in the MLPs
        C.rope = False  # whether to use Rotary Positional Embeddings in attention
        C.rms_norm = False  # whether to use RMSNorm instead of LayerNorm
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

        if train_config.cos_scheduler and train_config.lw_scheduler:
            raise ValueError("Cannot use both cosine scheduler and linear warmup scheduler simultaneously.")

        if train_config.lw_scheduler:
            scheduler = LinearWarmupScheduler(optimizer, warmup_steps=train_config.warmup_steps)

            return optimizer, scheduler
        
        if train_config.cos_scheduler:
            total_steps = train_config.max_iters - train_config.warmup_steps
            scheduler = CosineAnnealingWarmupScheduler(
                optimizer,
                warmup_steps=train_config.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=1e-4
            )
            return optimizer, scheduler

        return optimizer, None

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
# XXX: dataset copy cmd: cp /nobackup/archive/usr/dw87/pile_data_10.jsonl /nobackup/autodelete/usr/rsinema

if __name__ == '__main__':
    # # NewGELU test
    # gelu = NewGELU()
    # x = torch.linspace(-3, 3, steps=100).unsqueeze(0)  # add batch dimension
    # print(f"X shape: {x.shape}, GELU(X) shape: {gelu(x).shape}")

    # # SwiGLU test
    # swiglu = SwiGLU(dim_in=100, dim_out=100)
    # x = torch.linspace(-3, 3, steps=100).unsqueeze(0)  # add batch dimension
    # print(f"X shape: {x.shape}, SwiGLU(X) shape: {swiglu(x).shape}")

    # # Block with SwiGLU test
    # config = GPT.get_default_config()
    # config.n_embd = 100
    # config.n_head = 5
    # config.n_layer = 2
    # config.block_size = 10
    # config.vocab_size = 1000
    # config.swiglu = True  # Enable SwiGLU activations
    # block = Block(config)
    # x = torch.randn(2, 10, 100)  # batch size 2
    # print(f"Input shape: {x.shape}, Block output shape: {block(x).shape}")

    # # Rotary Embedding test
    # rotary_emb = RotaryEmbedding(dim=32)
    # seq_len = 20
    # cos, sin = rotary_emb(seq_len)
    # print(f"Cosine shape: {cos.shape}, Sine shape: {sin.shape}")

    # # CausalSelfAttention with RoPE test
    # config = GPT.get_default_config()
    # config.n_embd = 64
    # config.n_head = 8
    # config.n_layer = 2
    # config.block_size = 20
    # config.vocab_size = 1000
    # config.rope = True  # Enable RoPE
    # attn = CausalSelfAttention(config)
    # x = torch.randn(2, 20, 64)  # batch size, sequence length, embedding dim
    # print(f"Input shape: {x.shape}, CausalSelfAttention output shape: {attn(x).shape}")

    # RMSNorm test
    config = GPT.get_default_config()
    config.n_embd = 64
    config.n_head = 8
    config.n_layer = 2
    config.block_size = 20
    config.rms_norm = True  # Enable RMSNorm
    # Block with RMSNorm test
    block = Block(config)
    x = torch.randn(2, 20, 64)  # batch size, sequence length, embedding dim
    print(f"Input shape: {x.shape}, Block with RMSNorm output shape: {block(x).shape}")