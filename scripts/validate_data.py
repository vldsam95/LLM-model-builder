#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import dataset_stats
from src.data.validators import validate_jsonl_dialogues, validate_text_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw dataset files.")
    parser.add_argument("--raw-dir", default="data/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    checks = {
        f"{raw_dir}/pretrain_en.txt": validate_text_file(f"{raw_dir}/pretrain_en.txt"),
        f"{raw_dir}/pretrain_he.txt": validate_text_file(f"{raw_dir}/pretrain_he.txt"),
        f"{raw_dir}/domain_notes.txt": validate_text_file(f"{raw_dir}/domain_notes.txt"),
        f"{raw_dir}/sft_dialogues.jsonl": validate_jsonl_dialogues(f"{raw_dir}/sft_dialogues.jsonl"),
    }
    stats = {
        "pretrain_en": dataset_stats(f"{raw_dir}/pretrain_en.txt"),
        "pretrain_he": dataset_stats(f"{raw_dir}/pretrain_he.txt"),
        "domain_notes": dataset_stats(f"{raw_dir}/domain_notes.txt"),
        "sft_dialogues": dataset_stats(f"{raw_dir}/sft_dialogues.jsonl"),
    }
    print(json.dumps({"issues": checks, "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
