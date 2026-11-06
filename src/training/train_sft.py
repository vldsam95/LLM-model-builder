from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import DialogueDataset
from src.model.config import ModelConfig
from src.model.transformer import TinyGPT
from src.training.checkpointing import load_model_state_flexible
from src.training.loops import train_model
from src.training.optim import cosine_with_warmup, create_optimizer
from src.utils.config_loader import load_yaml
from src.utils.logging_utils import configure_logging
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the tiny GPT model on supervised dialogue data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))

    device = config.get("device", "cpu")
    logger = configure_logging("logs", config["experiment_name"])
    model_config = ModelConfig.from_dict(config["model"])
    model = TinyGPT(model_config).to(device)

    checkpoint_path = config.get("init_checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state_flexible(model, checkpoint["model_state"])
        logger.info("Loaded init checkpoint %s", checkpoint_path)

    train_dataset = DialogueDataset(config["train_sft_path"])
    val_dataset = DialogueDataset(config["val_sft_path"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    optimizer = create_optimizer(model, config["learning_rate"], config["weight_decay"])
    total_steps = max(1, len(train_loader) * config["num_epochs"] // max(1, config["grad_accum_steps"]))
    min_lr_scale = config["min_learning_rate"] / config["learning_rate"]
    scheduler = cosine_with_warmup(optimizer, config["warmup_steps"], total_steps, min_lr_scale=min_lr_scale)

    logger.info("Parameter count: %s", f"{model.num_parameters():,}")
    results = train_model(model, train_loader, val_loader, optimizer, scheduler, config, device, logger)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    with (Path(config["output_dir"]) / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
