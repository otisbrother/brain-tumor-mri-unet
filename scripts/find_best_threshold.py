#!/usr/bin/env python
"""Select a probability threshold by mean patient Dice on validation data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import BraTSVolumeDataset  # noqa: E402
from src.evaluation.metrics import binary_segmentation_metrics  # noqa: E402
from src.inference.predict import load_model_checkpoint, predict_volume_probabilities  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=np.arange(0.30, 0.71, 0.05).round(2).tolist())
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/threshold_search.json"))
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write the selected validation threshold back to the supplied config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    model, _ = load_model_checkpoint(config, args.checkpoint, device)
    dataset = BraTSVolumeDataset(
        config["data"]["metadata_csv"], Path(config["data"]["splits_dir"]) / "val.txt", config["data"]["modalities"]
    )
    scores = {float(threshold): [] for threshold in args.thresholds}
    for sample in tqdm(dataset, desc="Searching validation threshold", dynamic_ncols=True):
        probabilities = predict_volume_probabilities(
            model,
            sample["image"],
            int(config["data"]["target_size"]),
            int(config["inference"]["batch_size"]),
            device,
        )
        for threshold in scores:
            scores[threshold].append(binary_segmentation_metrics(probabilities, sample["mask"], threshold=threshold)["dice"])
    mean_scores = {f"{threshold:.2f}": float(np.mean(values)) for threshold, values in scores.items()}
    best = max(scores, key=lambda threshold: (np.mean(scores[threshold]), -abs(threshold - 0.5)))
    output = {"split": "val", "best_threshold": best, "best_mean_dice": float(np.mean(scores[best])), "mean_dice_by_threshold": mean_scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.update_config:
        config["inference"]["threshold"] = float(best)
        args.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"Updated inference.threshold in {args.config}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
