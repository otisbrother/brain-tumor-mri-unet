"""Evaluate a checkpoint on internal test patients with complete ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import BraTSVolumeDataset
from src.data.nifti import save_nifti
from src.evaluation.metrics import binary_segmentation_metrics
from src.inference.predict import load_model_checkpoint, predict_volume
from src.utils.config import ensure_output_directories, load_config
from src.utils.device import get_device
from src.utils.seed import set_seed

METRIC_COLUMNS = ("dice", "iou", "precision", "recall", "sensitivity", "specificity", "accuracy")


def evaluate_dataset(
    config: dict,
    checkpoint_path: str | Path,
    split_name: str = "test",
    save_predictions: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run patient-wise full-volume evaluation and return aggregate results."""
    device = get_device()
    model, _ = load_model_checkpoint(config, checkpoint_path, device)
    threshold = float(config["inference"]["threshold"])
    dataset = BraTSVolumeDataset(
        config["data"]["metadata_csv"],
        Path(config["data"]["splits_dir"]) / f"{split_name}.txt",
        config["data"]["modalities"],
    )
    prediction_dir = Path(config["paths"]["output_dir"]) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for sample in tqdm(dataset, desc=f"Evaluating {split_name} patients"):
        result = predict_volume(model, sample["image"], sample["spacing"], config, device)
        if save_predictions:
            save_nifti(
                result["mask"],
                sample["affine"],
                sample["header"],
                prediction_dir / f"{sample['patient_id']}_pred_seg.nii.gz",
            )
        metrics = binary_segmentation_metrics(result["mask"], sample["mask"])
        rows.append({"patient_id": sample["patient_id"], **metrics, "predicted_tumor_voxels": result["tumor_voxels"]})
    per_patient = pd.DataFrame(rows)
    summary = {metric: float(per_patient[metric].mean()) for metric in METRIC_COLUMNS}
    summary.update(patient_count=len(per_patient), threshold=threshold, split=split_name)
    return summary, per_patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Compute metrics without writing every predicted NIfTI volume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    ensure_output_directories(config)
    summary, per_patient = evaluate_dataset(
        config,
        args.checkpoint,
        args.split,
        save_predictions=not args.no_save_predictions,
    )
    output_dir = Path(config["paths"]["output_dir"]) / "evaluation"
    per_patient.to_csv(output_dir / "per_patient.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
