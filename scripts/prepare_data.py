#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import build_sft_tensors, combine_text_files, encode_pretrain_corpus
from src.data.source_download import download_all_data
from tokenizer.tokenizer_utils import SentencePieceTokenizer, train_sentencepiece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tokenizer and processed tensors for training.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--tokenizer-prefix", default="tokenizer/tiny_bilingual")
    parser.add_argument("--vocab-size", type=int, default=3200)
    parser.add_argument("--sequence-length", type=int, default=192)
    parser.add_argument("--download", action="store_true", help="Download raw data before processing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_all_data(raw_dir, seed=args.seed)

    tokenizer_corpus = processed_dir / "tokenizer_corpus.txt"
    pretrain_all = processed_dir / "pretrain_all.txt"
    domain_corpus = processed_dir / "domain_corpus.txt"

    combine_text_files(
        [
            raw_dir / "pretrain_en.txt",
            raw_dir / "pretrain_he.txt",
            raw_dir / "domain_notes.txt",
            raw_dir / "sft_dialogues.jsonl",
        ],
        tokenizer_corpus,
    )
    combine_text_files([raw_dir / "pretrain_en.txt", raw_dir / "pretrain_he.txt"], pretrain_all)
    combine_text_files([raw_dir / "domain_notes.txt"], domain_corpus)

    train_sentencepiece(tokenizer_corpus, args.tokenizer_prefix, vocab_size=args.vocab_size)
    tokenizer = SentencePieceTokenizer(f"{args.tokenizer_prefix}.model")

    stats = {
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "pretrain": encode_pretrain_corpus(
            tokenizer,
            pretrain_all,
            processed_dir / "pretrain_train_tokens.pt",
            processed_dir / "pretrain_val_tokens.pt",
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
        "domain": encode_pretrain_corpus(
            tokenizer,
            domain_corpus,
            processed_dir / "domain_train_tokens.pt",
            processed_dir / "domain_val_tokens.pt",
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
        "sft": build_sft_tensors(
            tokenizer,
            raw_dir / "sft_dialogues.jsonl",
            processed_dir / "sft_train.pt",
            processed_dir / "sft_val.pt",
            max_length=args.sequence_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
    }
    (processed_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
