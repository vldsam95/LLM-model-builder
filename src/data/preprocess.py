from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch

from src.data.dialogue_format import DialogueExample, build_dialogue_example
from tokenizer.tokenizer_utils import SentencePieceTokenizer


SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def split_long_text(text: str, max_chars: int = 700) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:max_chars]]


def read_nonempty_lines(path: str | Path, split_long_lines: bool = True) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            cleaned = normalize_text(raw)
            if not cleaned:
                continue
            if split_long_lines:
                rows.extend(split_long_text(cleaned))
            else:
                rows.append(cleaned)
    return rows


def combine_text_files(paths: list[str | Path], output_path: str | Path) -> int:
    lines: list[str] = []
    for path in paths:
        if Path(path).exists():
            split_long_lines = Path(path).suffix != ".jsonl"
            lines.extend(read_nonempty_lines(path, split_long_lines=split_long_lines))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def split_token_tensor(tokens: list[int], val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) < 10:
        raise ValueError("Not enough tokens to split.")
    split_idx = int(len(tokens) * (1 - val_ratio))
    split_idx = max(1, min(split_idx, len(tokens) - 1))
    train_ids = torch.tensor(tokens[:split_idx], dtype=torch.long)
    val_ids = torch.tensor(tokens[split_idx:], dtype=torch.long)
    return train_ids, val_ids


def encode_pretrain_corpus(
    tokenizer: SentencePieceTokenizer,
    input_path: str | Path,
    train_out: str | Path,
    val_out: str | Path,
    val_ratio: float,
    seed: int,
) -> dict[str, int]:
    text = Path(input_path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_ids, val_ids = split_token_tensor(tokens, val_ratio=val_ratio, seed=seed)
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_ids, train_out)
    torch.save(val_ids, val_out)
    return {"total_tokens": len(tokens), "train_tokens": len(train_ids), "val_tokens": len(val_ids)}


def load_dialogue_jsonl(path: str | Path) -> list[DialogueExample]:
    rows: list[DialogueExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            rows.append(
                DialogueExample(
                    system=obj.get("system"),
                    user=normalize_text(obj["user"]),
                    assistant=normalize_text(obj["assistant"]),
                )
            )
    return rows


def build_sft_tensors(
    tokenizer: SentencePieceTokenizer,
    input_path: str | Path,
    train_out: str | Path,
    val_out: str | Path,
    max_length: int,
    val_ratio: float,
    seed: int,
) -> dict[str, int]:
    examples = load_dialogue_jsonl(input_path)
    rng = random.Random(seed)
    rng.shuffle(examples)

    split_idx = int(len(examples) * (1 - val_ratio))
    split_idx = max(1, min(split_idx, len(examples) - 1))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    def encode_rows(rows: list[DialogueExample]) -> list[dict[str, torch.Tensor]]:
        encoded: list[dict[str, torch.Tensor]] = []
        for row in rows:
            input_ids, labels = build_dialogue_example(tokenizer, row, max_length=max_length)
            encoded.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        return encoded

    torch.save(encode_rows(train_examples), train_out)
    torch.save(encode_rows(val_examples), val_out)
    return {"train_examples": len(train_examples), "val_examples": len(val_examples)}


def dataset_stats(path: str | Path) -> dict[str, int]:
    path = Path(path)
    stats = {"bytes": int(path.stat().st_size if path.exists() else 0)}
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            stats["rows"] = sum(1 for line in handle if line.strip())
    else:
        stats["rows"] = len(read_nonempty_lines(path)) if path.exists() else 0
    return stats
