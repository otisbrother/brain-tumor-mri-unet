"""Reusable epoch loops, early stopping, and structured checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import metrics_from_counts


@dataclass
class EarlyStopping:
    """Track validation Dice improvements without owning training policy."""

    patience: int
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def update(self, score: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = 0
    totals = torch.zeros(4, dtype=torch.int64, device=device)
    context = torch.enable_grad if training else torch.inference_mode
    description = "Train" if training else "Validation"
    with context():
        progress = tqdm(loader, desc=description, dynamic_ncols=True, mininterval=1.0, leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = mixed_precision and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = int(images.shape[0])
            total_loss += loss.detach().float() * batch_size
            sample_count += batch_size
            # Keep metric work on the accelerator. The previous implementation
            # copied full Bx1xHxW tensors to CPU on every batch, forcing a costly
            # CUDA synchronization and PCIe transfer.
            with torch.no_grad():
                predicted = torch.sigmoid(logits.detach()) >= threshold
                expected = masks >= 0.5
                totals += torch.stack(
                    (
                        torch.count_nonzero(predicted & expected),
                        torch.count_nonzero(predicted & ~expected),
                        torch.count_nonzero(~predicted & expected),
                        torch.count_nonzero(~predicted & ~expected),
                    )
                )
            if batch_index == 1 or batch_index % 50 == 0:
                progress.set_postfix(loss=f"{float(total_loss / sample_count):.4f}")
    if sample_count == 0:
        raise ValueError("DataLoader yielded no samples.")
    tp, fp, fn, tn = (int(value) for value in totals.cpu().tolist())
    mean_loss = float((total_loss / sample_count).cpu())
    return {
        "loss": mean_loss,
        **metrics_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn}),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    scaler: torch.amp.GradScaler | None,
    mixed_precision: bool,
) -> dict[str, float]:
    """Train for one epoch and return loss plus pixel-level metrics."""
    return _run_epoch(model, loader, criterion, device, threshold, optimizer, scaler, mixed_precision)


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    mixed_precision: bool,
) -> dict[str, float]:
    """Evaluate one validation epoch without gradients or augmentation."""
    return _run_epoch(model, loader, criterion, device, threshold, mixed_precision=mixed_precision)


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_dice: float,
    config: dict[str, Any],
    threshold: float,
    *,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping_bad_epochs: int = 0,
    history: list[dict[str, float | int]] | None = None,
    pending_train_metrics: dict[str, float] | None = None,
) -> Path:
    """Atomically save all state needed to reproduce or resume an experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    unwrapped_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "best_val_dice": float(best_val_dice),
            "early_stopping_bad_epochs": int(early_stopping_bad_epochs),
            "threshold": float(threshold),
            "config": config,
            "history": list(history or []),
            "pending_train_metrics": pending_train_metrics,
        },
        temporary,
    )
    temporary.replace(destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, Any]:
    """Restore model and training state from a structured checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Training checkpoint must contain model_state_dict.")
    unwrapped_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    unwrapped_model.load_state_dict(state_dict)
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
