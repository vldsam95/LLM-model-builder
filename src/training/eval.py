from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inference.chat_engine import ChatEngine


DEFAULT_PROMPTS = [
    {"system": "Answer briefly in English.", "user": "Hello"},
    {"system": "Answer briefly in Hebrew.", "user": "מה זה SEO?"},
    {"system": "Act as a translation assistant.", "user": "Translate 'house' to Hebrew"},
    {"system": "Respond in compact JSON.", "user": "Explain AI in JSON"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare outputs from several checkpoints.")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece model path.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to compare.")
    parser.add_argument("--output", default="logs/eval_compare.json", help="Where to save evaluation results.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", dest="top_k", type=int, default=20)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, list[dict[str, str]]] = {}

    for checkpoint in args.checkpoints:
        if not Path(checkpoint).exists():
            continue
        engine = ChatEngine(
            checkpoint_path=checkpoint,
            tokenizer_path=args.tokenizer,
            device="cpu",
        )
        rows: list[dict[str, str]] = []
        for prompt in DEFAULT_PROMPTS:
            response = engine.chat(
                user_text=prompt["user"],
                system_text=prompt["system"],
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append({"system": prompt["system"], "user": prompt["user"], "assistant": response})
        results[checkpoint] = rows

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
