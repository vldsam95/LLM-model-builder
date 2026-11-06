#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builder.service import (
    DEFAULT_TOKENIZER,
    MODEL_PRESETS,
    NEUTRAL_FALLBACK_DIALOGUES,
    effective_default_base_checkpoint,
    load_project_manifest,
    project_dir,
    resolve_project_tokenizer,
    runtime_path,
    utc_now,
    write_builder_config,
    write_json,
)
from src.data.preprocess import build_sft_tensors, combine_text_files, encode_pretrain_corpus
from tokenizer.tokenizer_utils import SentencePieceTokenizer, train_sentencepiece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a builder project training job.")
    parser.add_argument("--project-slug", required=True)
    return parser.parse_args()


def venv_python() -> str:
    return str(PROJECT_ROOT / ".venv" / "bin" / "python")


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def model_preset(manifest: dict) -> dict:
    return MODEL_PRESETS.get(manifest.get("model_size", "safe_6m"), MODEL_PRESETS["safe_6m"])


def base_model_config(manifest: dict, vocab_size: int, sequence_length: int, dropout: float | None = None) -> dict:
    preset = model_preset(manifest)
    config = dict(preset["config"])
    config["vocab_size"] = vocab_size
    config["max_seq_len"] = sequence_length
    if dropout is not None:
        config["dropout"] = dropout
    return config


def checkpoint_model_config(checkpoint_rel: str) -> tuple[dict, dict]:
    checkpoint = torch.load(PROJECT_ROOT / checkpoint_rel, map_location="cpu")
    return checkpoint["config"]["model"], checkpoint["config"]


def merge_jsonl_files(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    key = (obj.get("system", "") or "", obj.get("user", "") or "", obj.get("assistant", "") or "")
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def foundation_pretrain_paths(manifest: dict) -> list[Path]:
    if manifest.get("foundation_pack", "none") != "general_open_pack":
        return []
    return [
        PROJECT_ROOT / "data" / "raw" / "pretrain_en.txt",
        PROJECT_ROOT / "data" / "raw" / "pretrain_he.txt",
    ]


def foundation_sft_path(manifest: dict) -> Path | None:
    if manifest.get("foundation_pack", "none") != "general_open_pack":
        return None
    path = PROJECT_ROOT / NEUTRAL_FALLBACK_DIALOGUES
    return path if path.exists() else None


def inflate_small_corpus(path: Path, min_chars: int) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if len(text) >= min_chars:
        return
    repeats = max(2, (min_chars // max(1, len(text))) + 1)
    path.write_text("\n".join(text for _ in range(repeats)), encoding="utf-8")


def fast_adapter_config(slug: str, manifest: dict, train_path: Path, val_path: Path) -> dict:
    base_checkpoint = manifest.get("base_checkpoint") or effective_default_base_checkpoint()
    model_config, checkpoint_config = checkpoint_model_config(base_checkpoint)
    sequence_length = min(int(manifest.get("sequence_length", model_config["max_seq_len"])), int(model_config["max_seq_len"]))
    return {
        "seed": 42,
        "device": "cpu",
        "experiment_name": f"builder_{slug}_fast_adapter",
        "output_dir": str(project_dir(slug) / "checkpoints" / "fast_adapter"),
        "tokenizer_model": manifest.get("tokenizer_model", checkpoint_config.get("tokenizer_model", DEFAULT_TOKENIZER)),
        "train_sft_path": str(train_path),
        "val_sft_path": str(val_path),
        "sequence_length": sequence_length,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "num_epochs": 8,
        "learning_rate": 0.00015,
        "min_learning_rate": 0.000015,
        "weight_decay": 0.01,
        "dropout": 0.05,
        "max_grad_norm": 1.0,
        "eval_every_steps": 20,
        "save_every_steps": 40,
        "log_every_steps": 10,
        "warmup_steps": 10,
        "early_stopping_patience": 4,
        "resume_from": None,
        "init_checkpoint": base_checkpoint,
        "num_workers": 0,
        "amp": False,
        "model": {**model_config, "max_seq_len": sequence_length},
    }


def pretrain_config(
    slug: str,
    manifest: dict,
    tokenizer_model: str,
    train_tokens: Path,
    val_tokens: Path,
    output_dir: Path,
    sequence_length: int,
    vocab_size: int,
    init_checkpoint: str | None = None,
    num_epochs: int = 2,
    learning_rate: float = 0.0006,
) -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "experiment_name": f"builder_{slug}_{output_dir.name}",
        "output_dir": str(output_dir),
        "tokenizer_model": tokenizer_model,
        "train_tokens_path": str(train_tokens),
        "val_tokens_path": str(val_tokens),
        "sequence_length": sequence_length,
        "batch_size": 8 if sequence_length <= 128 else 6,
        "grad_accum_steps": 2,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "min_learning_rate": learning_rate / 10,
        "weight_decay": 0.05,
        "dropout": 0.1,
        "max_grad_norm": 1.0,
        "eval_every_steps": 50,
        "save_every_steps": 100,
        "log_every_steps": 10,
        "warmup_steps": 20,
        "early_stopping_patience": 3,
        "resume_from": None,
        "init_checkpoint": init_checkpoint,
        "num_workers": 0,
        "amp": False,
        "model": base_model_config(manifest, vocab_size=vocab_size, sequence_length=sequence_length, dropout=0.1),
    }


def sft_from_scratch_config(
    slug: str,
    manifest: dict,
    tokenizer_model: str,
    train_path: Path,
    val_path: Path,
    init_checkpoint: str,
    sequence_length: int,
    vocab_size: int,
) -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "experiment_name": f"builder_{slug}_sft",
        "output_dir": str(project_dir(slug) / "checkpoints" / "sft"),
        "tokenizer_model": tokenizer_model,
        "train_sft_path": str(train_path),
        "val_sft_path": str(val_path),
        "sequence_length": sequence_length,
        "batch_size": 8 if sequence_length <= 128 else 6,
        "grad_accum_steps": 1,
        "num_epochs": 8,
        "learning_rate": 0.0002,
        "min_learning_rate": 0.00002,
        "weight_decay": 0.01,
        "dropout": 0.05,
        "max_grad_norm": 1.0,
        "eval_every_steps": 20,
        "save_every_steps": 40,
        "log_every_steps": 10,
        "warmup_steps": 10,
        "early_stopping_patience": 4,
        "resume_from": None,
        "init_checkpoint": init_checkpoint,
        "num_workers": 0,
        "amp": False,
        "model": base_model_config(manifest, vocab_size=vocab_size, sequence_length=sequence_length, dropout=0.05),
    }


def run_fast_adapter(slug: str, manifest: dict) -> tuple[str, str]:
    root = project_dir(slug)
    base_checkpoint = manifest.get("base_checkpoint") or effective_default_base_checkpoint()
    _, checkpoint_config = checkpoint_model_config(base_checkpoint)
    tokenizer_rel = checkpoint_config.get("tokenizer_model", manifest.get("tokenizer_model", DEFAULT_TOKENIZER))
    tokenizer_path = PROJECT_ROOT / tokenizer_rel
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    train_path = root / "processed" / "sft_train.pt"
    val_path = root / "processed" / "sft_val.pt"
    sft_source = root / "processed" / "sft_source.jsonl"
    sft_paths = [root / "raw" / "sft_dialogues.jsonl"]
    foundation_sft = foundation_sft_path(manifest)
    if foundation_sft:
        sft_paths.insert(0, foundation_sft)
    merge_jsonl_files(sft_paths, sft_source)
    max_length = min(
        int(manifest.get("sequence_length", 128)),
        int(checkpoint_model_config(base_checkpoint)[0]["max_seq_len"]),
    )
    build_sft_tensors(
        tokenizer=tokenizer,
        input_path=sft_source,
        train_out=train_path,
        val_out=val_path,
        max_length=max_length,
        val_ratio=0.1,
        seed=42,
    )
    config_path = root / "configs" / "fast_adapter.yaml"
    write_builder_config(config_path, fast_adapter_config(slug, manifest, train_path, val_path))
    run_command([venv_python(), "-m", "src.training.train_sft", "--config", str(config_path)])
    return str((root / "checkpoints" / "fast_adapter" / "best.pt").relative_to(PROJECT_ROOT)), tokenizer_rel


def run_from_scratch(slug: str, manifest: dict) -> tuple[str, str]:
    root = project_dir(slug)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    tokenizer_dir = root / "tokenizer"
    tokenizer_prefix = tokenizer_dir / "project"
    preset = model_preset(manifest)
    sequence_length = min(int(manifest.get("sequence_length", preset["sequence_length"])), int(preset["sequence_length"]))
    vocab_size = int(manifest.get("vocab_size", preset["vocab_size"]) or preset["vocab_size"])

    tokenizer_corpus = processed_dir / "tokenizer_corpus.txt"
    pretrain_all = processed_dir / "pretrain_all.txt"
    domain_corpus = processed_dir / "domain_corpus.txt"
    sft_source = processed_dir / "sft_source.jsonl"
    sft_paths = [raw_dir / "sft_dialogues.jsonl"]
    foundation_sft = foundation_sft_path(manifest)
    if foundation_sft:
        sft_paths.insert(0, foundation_sft)
    merge_jsonl_files(sft_paths, sft_source)

    tokenizer_inputs = [raw_dir / "pretrain_en.txt", raw_dir / "pretrain_he.txt", raw_dir / "domain_notes.txt", sft_source]
    pretrain_inputs = [raw_dir / "pretrain_en.txt", raw_dir / "pretrain_he.txt"]
    foundation_pretrain = foundation_pretrain_paths(manifest)
    tokenizer_inputs.extend(foundation_pretrain)
    pretrain_inputs.extend(foundation_pretrain)

    combine_text_files(tokenizer_inputs, tokenizer_corpus)
    combine_text_files(pretrain_inputs, pretrain_all)
    combine_text_files([raw_dir / "domain_notes.txt"], domain_corpus)

    if not pretrain_all.exists() or not pretrain_all.read_text(encoding="utf-8").strip():
        pretrain_all.write_text((raw_dir / "source_text.txt").read_text(encoding="utf-8"), encoding="utf-8")
    if not domain_corpus.exists() or not domain_corpus.read_text(encoding="utf-8").strip():
        domain_corpus.write_text(pretrain_all.read_text(encoding="utf-8"), encoding="utf-8")

    min_corpus_chars = max(3000, sequence_length * 24)
    inflate_small_corpus(pretrain_all, min_chars=min_corpus_chars)
    inflate_small_corpus(domain_corpus, min_chars=min_corpus_chars)

    train_sentencepiece(tokenizer_corpus, tokenizer_prefix, vocab_size=vocab_size)
    tokenizer_model = tokenizer_prefix.with_suffix(".model")
    tokenizer = SentencePieceTokenizer(tokenizer_model)

    pretrain_train = processed_dir / "pretrain_train_tokens.pt"
    pretrain_val = processed_dir / "pretrain_val_tokens.pt"
    domain_train = processed_dir / "domain_train_tokens.pt"
    domain_val = processed_dir / "domain_val_tokens.pt"
    sft_train = processed_dir / "sft_train.pt"
    sft_val = processed_dir / "sft_val.pt"

    encode_pretrain_corpus(tokenizer, pretrain_all, pretrain_train, pretrain_val, val_ratio=0.1, seed=42)
    encode_pretrain_corpus(tokenizer, domain_corpus, domain_train, domain_val, val_ratio=0.1, seed=42)
    build_sft_tensors(tokenizer, sft_source, sft_train, sft_val, max_length=sequence_length, val_ratio=0.1, seed=42)

    pretrain_config_path = root / "configs" / "pretrain.yaml"
    domain_config_path = root / "configs" / "domain.yaml"
    sft_config_path = root / "configs" / "sft.yaml"

    pretrain_output = root / "checkpoints" / "pretrain"
    domain_output = root / "checkpoints" / "domain"
    sft_output = root / "checkpoints" / "sft"
    tokenizer_rel = str(tokenizer_model.relative_to(PROJECT_ROOT))

    write_builder_config(
        pretrain_config_path,
        pretrain_config(
            slug,
            manifest,
            tokenizer_rel,
            pretrain_train,
            pretrain_val,
            pretrain_output,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            init_checkpoint=None,
            num_epochs=2,
            learning_rate=0.0006,
        ),
    )
    run_command([venv_python(), "-m", "src.training.train_pretrain", "--config", str(pretrain_config_path)])

    write_builder_config(
        domain_config_path,
        pretrain_config(
            slug,
            manifest,
            tokenizer_rel,
            domain_train,
            domain_val,
            domain_output,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            init_checkpoint=str((pretrain_output / "best.pt").relative_to(PROJECT_ROOT)),
            num_epochs=2,
            learning_rate=0.0003,
        ),
    )
    run_command([venv_python(), "-m", "src.training.train_pretrain", "--config", str(domain_config_path)])

    write_builder_config(
        sft_config_path,
        sft_from_scratch_config(
            slug,
            manifest,
            tokenizer_rel,
            sft_train,
            sft_val,
            init_checkpoint=str((domain_output / "best.pt").relative_to(PROJECT_ROOT)),
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        ),
    )
    run_command([venv_python(), "-m", "src.training.train_sft", "--config", str(sft_config_path)])
    return str((sft_output / "best.pt").relative_to(PROJECT_ROOT)), tokenizer_rel


def update_runtime_success(slug: str, checkpoint_rel: str, tokenizer_rel: str) -> None:
    runtime = {
        "status": "completed",
        "pid": None,
        "started_at": None,
        "completed_at": utc_now(),
        "error": None,
        "log_path": str(project_dir(slug) / "logs" / "train.log"),
        "active_checkpoint": checkpoint_rel,
        "active_tokenizer": tokenizer_rel,
    }
    write_json(runtime_path(slug), runtime)


def update_runtime_failure(slug: str, error_text: str) -> None:
    runtime = {
        "status": "failed",
        "pid": None,
        "started_at": None,
        "completed_at": utc_now(),
        "error": error_text,
        "log_path": str(project_dir(slug) / "logs" / "train.log"),
        "active_checkpoint": load_project_manifest(slug).get("base_checkpoint", effective_default_base_checkpoint()),
        "active_tokenizer": resolve_project_tokenizer(slug),
    }
    write_json(runtime_path(slug), runtime)


def main() -> None:
    args = parse_args()
    manifest = load_project_manifest(args.project_slug)
    try:
        if manifest.get("build_mode") == "from_scratch_tiny":
            checkpoint_rel, tokenizer_rel = run_from_scratch(args.project_slug, manifest)
        else:
            checkpoint_rel, tokenizer_rel = run_fast_adapter(args.project_slug, manifest)
        update_runtime_success(args.project_slug, checkpoint_rel, tokenizer_rel)
    except Exception:
        update_runtime_failure(args.project_slug, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
