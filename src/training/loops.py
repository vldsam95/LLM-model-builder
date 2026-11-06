from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.checkpointing import save_checkpoint


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    losses: list[float] = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        output = model(batch["input_ids"], labels=batch["labels"])
        losses.append(output["loss"].item())
    return sum(losses) / max(1, len(losses))


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict[str, Any],
    device: str,
    logger,
) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    epochs_without_improvement = 0

    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    num_epochs = int(config["num_epochs"])
    eval_every_steps = int(config["eval_every_steps"])
    save_every_steps = int(config["save_every_steps"])
    log_every_steps = int(config["log_every_steps"])
    max_grad_norm = float(config["max_grad_norm"])
    patience = int(config["early_stopping_patience"])

    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        logger.info("Resumed from %s", config["resume_from"])

    history: list[dict[str, float | int]] = []
    scaler = torch.amp.GradScaler(enabled=bool(config.get("amp", False) and device.startswith("cuda")))
    training_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.split(":")[0], enabled=bool(config.get("amp", False) and device.startswith("cuda"))):
                output = model(batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if batch_index % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1

                if global_step % log_every_steps == 0:
                    avg_train_loss = running_loss / batch_index
                    progress.set_postfix(loss=f"{avg_train_loss:.4f}", step=global_step)
                    logger.info(
                        "epoch=%s step=%s train_loss=%.4f lr=%.6f",
                        epoch + 1,
                        global_step,
                        avg_train_loss,
                        optimizer.param_groups[0]["lr"],
                    )

                if global_step % eval_every_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    history.append({"epoch": epoch + 1, "step": global_step, "val_loss": val_loss})
                    logger.info("epoch=%s step=%s val_loss=%.4f", epoch + 1, global_step, val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            config,
                            epoch + 1,
                            global_step,
                            best_val_loss,
                        )
                    else:
                        epochs_without_improvement += 1
                    model.train()

                if global_step % save_every_steps == 0:
                    save_checkpoint(
                        output_dir / f"step_{global_step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch + 1,
                        global_step,
                        best_val_loss,
                    )

        if len(train_loader) % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            global_step += 1

        epoch_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device)
        logger.info(
            "epoch=%s completed train_loss=%.4f val_loss=%.4f epoch_seconds=%.2f",
            epoch + 1,
            epoch_loss,
            val_loss,
            time.time() - epoch_start,
        )
        history.append({"epoch": epoch + 1, "step": global_step, "train_loss": epoch_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch + 1,
                global_step,
                best_val_loss,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after %s epochs without improvement.", patience)
                break

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        config,
        epoch + 1,
        global_step,
        best_val_loss,
    )
    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "training_seconds": time.time() - training_start,
    }
