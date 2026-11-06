from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    def __init__(self, tokens_path: str | Path, seq_len: int) -> None:
        self.tokens = torch.load(tokens_path, map_location="cpu")
        self.seq_len = seq_len
        if len(self.tokens) <= seq_len:
            raise ValueError(f"Token tensor in {tokens_path} is too short for seq_len={seq_len}")
        self.num_samples = max(1, (len(self.tokens) - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = random.randint(0, len(self.tokens) - self.seq_len - 2)
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


class DialogueDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.rows = torch.load(path, map_location="cpu")
        if not self.rows:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]
