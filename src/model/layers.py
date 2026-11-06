from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unsupported norm_type={norm_type}")


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int(config.d_model * config.ffn_mult)
        self.ffn_type = config.ffn_type
        self.dropout = nn.Dropout(config.dropout)
        if self.ffn_type == "gelu":
            self.up = nn.Linear(config.d_model, hidden, bias=config.bias)
            self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
            self.act = nn.GELU()
        elif self.ffn_type == "swiglu":
            self.up = nn.Linear(config.d_model, hidden, bias=config.bias)
            self.gate = nn.Linear(config.d_model, hidden, bias=config.bias)
            self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported ffn_type={self.ffn_type}")

    def forward(self, x: Tensor) -> Tensor:
        if self.ffn_type == "gelu":
            x = self.down(self.act(self.up(x)))
        else:
            x = self.down(self.act(self.gate(x)) * self.up(x))
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool(),
            persistent=False,
        )

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat_factor = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat_factor, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = build_norm(config.norm_type, config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config.norm_type, config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x
