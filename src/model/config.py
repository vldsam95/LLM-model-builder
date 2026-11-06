from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    ffn_mult: float = 4.0
    dropout: float = 0.1
    bias: bool = True
    norm_type: str = "layernorm"
    ffn_type: str = "gelu"
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)
