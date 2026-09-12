#!/usr/bin/env python
"""Create reproducible, non-overlapping patient-level train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


def split_patient_ids(
    patient_ids: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    """Shuffle and split patient IDs without rounding overlap or loss."""
    ratios = np.asarray((train_ratio, val_ratio, test_ratio), dtype=float)
    if len(patient_ids) < 3:
        raise ValueError("At least three patients are required for train/val/test splits.")
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Split ratios must be positive and sum to 1.0, got {ratios.tolist()}")
    unique = sorted(set(patient_id.lower() for patient_id in patient_ids))
    if len(unique) != len(patient_ids):
        raise ValueError("Metadata contains duplicate patient IDs.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_total = len(unique)
    n_train = int(np.floor(n_total * train_ratio))
    n_val = int(np.floor(n_total * val_ratio))
    if n_train == 0 or n_val == 0 or n_total - n_train - n_val == 0:
        raise ValueError("Dataset is too small for non-empty splits at the configured ratios.")
    return {
        "train": sorted(unique[:n_train]),
        "val": sorted(unique[n_train : n_train + n_val]),
        "test": sorted(unique[n_train + n_val :]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["data"]["splits_dir"])
    targets = {name: output_dir / f"{name}.txt" for name in SPLIT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"Split files already exist; refusing to regenerate: {existing}. Use --force intentionally.")
    metadata = pd.read_csv(config["data"]["metadata_csv"], dtype={"patient_id": str})
    splits = split_patient_ids(
        metadata["patient_id"].tolist(),
        float(config["data"]["train_ratio"]),
        float(config["data"]["val_ratio"]),
        float(config["data"]["test_ratio"]),
        int(config["project"]["seed"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, patient_ids in splits.items():
        targets[name].write_text("\n".join(patient_ids) + "\n", encoding="utf-8")
        print(f"{name}: {len(patient_ids)} patients -> {targets[name]}")
    assert not (set(splits["train"]) & set(splits["val"]) | set(splits["train"]) & set(splits["test"]) | set(splits["val"]) & set(splits["test"]))


if __name__ == "__main__":
    main()

