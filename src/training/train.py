"""Train 2D U-Net on patient-safe BraTS axial slices."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.augmentation import SynchronizedAugmentation
from src.data.dataset import BraTSSliceDataset, PatientGroupedSampler
from src.inference.predict import build_model
from src.training.losses import BCEDiceLoss
from src.training.trainer import (
    EarlyStopping,
    load_training_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from src.utils.config import ensure_output_directories, load_config
from src.utils.device import get_device
from src.utils.logging import configure_logging
from src.utils.seed import set_seed


def _dataset(config: dict, split: str, transform=None, max_patients: int | None = None) -> BraTSSliceDataset:
    data = config["data"]
    return BraTSSliceDataset(
        metadata_csv=data["metadata_csv"],
        split_file=Path(data["splits_dir"]) / f"{split}.txt",
        modalities=data["modalities"],
        target_size=int(data["target_size"]),
        split_name=split,
        keep_negative_slices=bool(data.get("keep_negative_slices", True)),
        negative_to_positive_ratio=float(data.get("negative_to_positive_ratio", 1.0)),
        transform=transform,
        seed=int(config["project"]["seed"]),
        cache_size=int(data.get("cache_size", 2)),
        max_patients=max_patients,
        index_cache_dir=data.get("index_cache_dir"),
    )


def _loader(dataset, config: dict, shuffle: bool, sampler=None) -> DataLoader:
    training = config["training"]
    workers = int(training["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(training.get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _balanced_overfit_indices(dataset: BraTSSliceDataset, sample_count: int = 5) -> list[int]:
    """Choose a tiny deterministic mixture of tumor-positive and negative slices."""
    positive = [
        index for index, key in enumerate(dataset.samples) if key in dataset.positive_sample_keys
    ]
    negative = [
        index for index, key in enumerate(dataset.samples) if key not in dataset.positive_sample_keys
    ]
    selected = positive[: min(3, sample_count)]
    selected.extend(negative[: sample_count - len(selected)])
    if len(selected) < sample_count:
        remaining = [index for index in range(len(dataset)) if index not in selected]
        selected.extend(remaining[: sample_count - len(selected)])
    return selected


def save_training_curves(history: pd.DataFrame, output_dir: Path) -> None:
    """Save real experiment curves; skip cleanly when matplotlib is unavailable."""
    cache_dir = Path("outputs/.matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib
        matplotlib.use("Agg")  # Saving reports must not require a desktop GUI.
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        (("train_loss", "val_loss"), "Loss", "loss.png"),
        (("val_dice",), "Validation Dice", "dice.png"),
        (("val_iou",), "Validation IoU", "iou.png"),
        (("val_recall",), "Validation Recall", "recall.png"),
    ]
    for columns, title, filename in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        for column in columns:
            axis.plot(history["epoch"], history[column], label=column)
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--overfit-small-batch", action="store_true", help="Sanity-check learning on at most five slices.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last/best structured checkpoint.")
    parser.add_argument("--epochs-this-run", type=int, default=None,
                        help="Stop cleanly after this many complete train+validation epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs_this_run is not None and args.epochs_this_run < 1:
        raise SystemExit("--epochs-this-run must be positive")
    config = load_config(args.config)
    ensure_output_directories(config)
    logger = configure_logging(Path(config["paths"]["output_dir"]) / "logs" / "training.log")
    seed = int(config["project"]["seed"])
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    device = get_device()
    logger.info("Device: %s", device)

    checkpoint_path = Path(config["paths"]["best_checkpoint"])
    last_checkpoint_path = Path(
        config["paths"].get("last_checkpoint", checkpoint_path.with_name("last_model.pt"))
    )
    epochs = int(config["training"]["epochs"])
    early_enabled = bool(config["training"].get("early_stopping", {}).get("enabled", True))
    if args.overfit_small_batch:
        train_dataset = _dataset(config, "train", transform=None, max_patients=1)
        selected_indices = _balanced_overfit_indices(train_dataset, min(5, len(train_dataset)))
        subset_size = len(selected_indices)
        train_dataset = Subset(train_dataset, selected_indices)
        val_dataset = train_dataset
        epochs = int(config["training"].get("overfit_steps", 200))
        early_enabled = False
        checkpoint_path = checkpoint_path.with_name("overfit_model.pt")
        last_checkpoint_path = last_checkpoint_path.with_name("overfit_last_model.pt")
        logger.warning("Synthetic-free sanity mode: repeatedly fitting %d real training slices.", subset_size)
        train_sampler = None
    else:
        augmentation = None
        if config.get("augmentation", {}).get("enabled", True):
            augmentation = SynchronizedAugmentation(config["augmentation"])
        train_dataset = _dataset(config, "train", augmentation)
        val_dataset = _dataset(config, "val", None)
        train_sampler = PatientGroupedSampler(train_dataset, seed=seed)
    train_loader = _loader(train_dataset, config, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = _loader(val_dataset, config, shuffle=False)
    logger.info(
        "Samples: train=%d (%d batches), val=%d (%d batches).",
        len(train_dataset),
        len(train_loader),
        len(val_dataset),
        len(val_loader),
    )

    model = build_model(config).to(device)
    if (
        bool(config["training"].get("multi_gpu", False))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        model = torch.nn.DataParallel(model)
        logger.info("DataParallel enabled across %d CUDA devices.", torch.cuda.device_count())
    elif bool(config["training"].get("multi_gpu", False)):
        logger.warning("multi_gpu requested, but fewer than two CUDA devices are available.")
    loss_config = config["loss"]
    criterion = BCEDiceLoss(loss_config["bce_weight"], loss_config["dice_weight"], loss_config["smooth"])
    optimizer_config = config["optimizer"]
    if optimizer_config.get("name", "adamw").lower() != "adamw":
        raise ValueError("V1 supports only optimizer.name=adamw.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimizer_config["learning_rate"]), weight_decay=float(optimizer_config["weight_decay"])
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(scheduler_config["factor"]), patience=int(scheduler_config["patience"])
    )
    amp_enabled = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    threshold = float(config["inference"]["threshold"])
    stopping = EarlyStopping(int(config["training"].get("early_stopping", {}).get("patience", 10)))
    history_rows: list[dict[str, float | int]] = []
    history_path = Path(config["paths"]["output_dir"]) / "logs" / "training_history.csv"
    start_epoch = 1
    pending_train_metrics = None
    if args.resume is not None:
        resumed = load_training_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        completed_epoch = int(resumed.get("epoch", 0))
        pending_train_metrics = resumed.get("pending_train_metrics")
        start_epoch = completed_epoch + 1
        stopping.best_score = float(resumed.get("best_val_dice", float("-inf")))
        stopping.bad_epochs = int(resumed.get("early_stopping_bad_epochs", 0))
        history_rows = list(resumed.get("history") or [])
        if not history_rows and history_path.is_file():
            history_rows = pd.read_csv(history_path).to_dict("records")
            history_rows = [row for row in history_rows if int(row["epoch"]) <= completed_epoch]
        if train_sampler is not None:
            train_sampler.epoch = completed_epoch
        logger.info(
            "Resumed %s at completed epoch %d; continuing from epoch %d.",
            args.resume,
            completed_epoch,
            start_epoch,
        )

    end_epoch = epochs if args.epochs_this_run is None else min(epochs, start_epoch + args.epochs_this_run - 1)
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_started = time.perf_counter()
            if epoch == start_epoch and pending_train_metrics is not None:
                train_metrics = pending_train_metrics
                if train_sampler is not None:
                    train_sampler.epoch = epoch
                logger.info("Epoch %d/%d: restored completed TRAIN; continuing VALIDATION", epoch, epochs)
            else:
                logger.info("Epoch %d/%d: TRAIN starting", epoch, epochs)
                train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, threshold, scaler, amp_enabled)
                pending_path = config["paths"].get("pending_checkpoint")
                if pending_path:
                    save_checkpoint(
                        pending_path, epoch - 1, model, optimizer, scheduler,
                        stopping.best_score, copy.deepcopy(config), threshold,
                        scaler=scaler, early_stopping_bad_epochs=stopping.bad_epochs,
                        history=history_rows, pending_train_metrics=train_metrics,
                    )
                    logger.info("Saved train phase before validation: %s", pending_path)
            logger.info("Epoch %d/%d: TRAIN complete; VALIDATION starting", epoch, epochs)
            val_metrics = validate_one_epoch(model, val_loader, criterion, device, threshold, amp_enabled)
            scheduler.step(val_metrics["dice"])
            improved, should_stop = stopping.update(val_metrics["dice"])
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history_rows.append(row)
            history = pd.DataFrame(history_rows)
            history.to_csv(history_path, index=False)
            logger.info("Epoch %03d | %s", epoch, json.dumps({key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()}))
            if improved:
                save_checkpoint(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    stopping.best_score,
                    copy.deepcopy(config),
                    threshold,
                    scaler=scaler,
                    early_stopping_bad_epochs=stopping.bad_epochs,
                    history=history_rows,
                )
                logger.info("Saved best checkpoint: %s", checkpoint_path)
            save_checkpoint(
                last_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                stopping.best_score,
                copy.deepcopy(config),
                threshold,
                scaler=scaler,
                early_stopping_bad_epochs=stopping.bad_epochs,
                history=history_rows,
            )
            logger.info("Saved resumable checkpoint: %s", last_checkpoint_path)
            if early_enabled and should_stop:
                logger.info("Early stopping after %d epochs without Dice improvement.", stopping.bad_epochs)
                break
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError("CUDA out of memory. Reduce training.batch_size, data.target_size, or model.base_channels.") from exc
    if history_rows:
        save_training_curves(pd.DataFrame(history_rows), Path(config["paths"]["output_dir"]) / "figures")
    logger.info("Training invocation finished. Last completed epoch: %d", int(history_rows[-1]["epoch"]) if history_rows else 0)


if __name__ == "__main__":
    main()
