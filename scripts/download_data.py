#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.source_download import download_all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small open bilingual datasets for the project.")
    parser.add_argument("--raw-dir", default="data/raw", help="Where to write downloaded raw files.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_all_data(args.raw_dir, seed=args.seed)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
