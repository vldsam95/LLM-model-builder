#!/usr/bin/env python3
"""Regenerate neutral SFT rows and tensors for configs/builder_base_sft_cpu.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_builder_base_sft import main as build_rows
from src.data.preprocess import build_sft_tensors
from src.data.source_download import write_source_manifest
from tokenizer.tokenizer_utils import SentencePieceTokenizer


def main() -> None:
    tokenizer_path = PROJECT_ROOT / "tokenizer/tiny_bilingual.model"
    if not tokenizer_path.exists():
        raise SystemExit("Run scripts/prepare_data.py first to generate the local tokenizer.")
    build_rows()
    write_source_manifest(PROJECT_ROOT / "data/raw/source_manifest.json")
    build_sft_tensors(
        SentencePieceTokenizer(tokenizer_path),
        PROJECT_ROOT / "data/raw/sft_builder_base.jsonl",
        PROJECT_ROOT / "data/processed/sft_builder_base_train.pt",
        PROJECT_ROOT / "data/processed/sft_builder_base_val.pt",
        max_length=128,
        val_ratio=0.1,
        seed=42,
    )


if __name__ == "__main__":
    main()
