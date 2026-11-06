from __future__ import annotations

import torch

from src.model.transformer import TinyGPT


@torch.no_grad()
def sample_tokens(
    model: TinyGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
