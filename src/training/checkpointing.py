from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict[str, Any],
    epoch: int,
    step: int,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "config": config,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_model_state_flexible(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    adapted_state = dict(checkpoint_state)

    pos_key = "position_embedding.weight"
    if pos_key in checkpoint_state and pos_key in model_state:
        source = checkpoint_state[pos_key]
        target = model_state[pos_key]
        if source.shape != target.shape:
            if source.shape[1] != target.shape[1]:
                raise RuntimeError(f"Cannot adapt {pos_key}: hidden size mismatch {source.shape} vs {target.shape}")
            new_pos = target.clone()
            copy_len = min(source.shape[0], target.shape[0])
            new_pos[:copy_len] = source[:copy_len]
            if target.shape[0] > source.shape[0]:
                last_row = source[-1:].expand(target.shape[0] - source.shape[0], -1)
                new_pos[copy_len:] = last_row
            adapted_state[pos_key] = new_pos

    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    allowed_missing = {pos_key}
    real_missing = [key for key in missing if key not in allowed_missing]
    if real_missing:
        raise RuntimeError(f"Missing checkpoint keys: {real_missing}")
