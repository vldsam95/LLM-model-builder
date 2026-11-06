from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.tokenizer_utils import train_sentencepiece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument("--input", required=True, help="Combined training text file.")
    parser.add_argument("--output-prefix", default="tokenizer/tiny_bilingual", help="Output prefix for SentencePiece.")
    parser.add_argument("--vocab-size", type=int, default=3200, help="Tokenizer vocabulary size.")
    parser.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"], help="SentencePiece model type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    train_sentencepiece(
        input_path=args.input,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
