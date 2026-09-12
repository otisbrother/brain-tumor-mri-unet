#!/usr/bin/env python
"""Finish training, threshold selection, test evaluation, and artifact packaging."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(command: list[str]) -> None:
    """Run one pipeline stage in the repository and stream its output."""
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def checkpoint_epoch(path: Path) -> int | None:
    """Read a structured checkpoint epoch without loading tensors onto a GPU."""
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "epoch" not in checkpoint:
        raise ValueError(f"Invalid structured checkpoint: {path}")
    return int(checkpoint["epoch"])


def package_results(config_path: Path, output_path: Path) -> Path:
    """Package the model, effective config, logs, figures, and evaluation reports."""
    candidates = [
        ROOT / "models" / "best_model.pt",
        ROOT / "models" / "last_model.pt",
    ]
    for folder in (
        ROOT / "outputs" / "evaluation",
        ROOT / "outputs" / "logs",
        ROOT / "outputs" / "figures",
    ):
        if folder.is_dir():
            candidates.extend(path for path in folder.rglob("*") if path.is_file())

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
        # The local Streamlit application reads configs/config.yaml.
        archive.write(config_path, "configs/config.yaml")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kaggle_fast.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/brain-tumor-results.zip"),
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Run training even when last_model.pt already reached the configured epoch count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    from src.utils.config import load_config

    config = load_config(config_path)
    required = [
        ROOT / config["data"]["metadata_csv"],
        ROOT / config["data"]["splits_dir"] / "train.txt",
        ROOT / config["data"]["splits_dir"] / "val.txt",
        ROOT / config["data"]["splits_dir"] / "test.txt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Dataset preparation is incomplete; missing: {missing}")

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for the fast completion pipeline.")

    best_path = ROOT / config["paths"]["best_checkpoint"]
    last_path = ROOT / config["paths"].get("last_checkpoint", "models/last_model.pt")
    target_epochs = int(config["training"]["epochs"])
    last_epoch = checkpoint_epoch(last_path)

    if args.force_train or last_epoch is None or last_epoch < target_epochs:
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.training.train",
            "--config",
            str(config_path),
        ]
        if last_epoch is not None:
            command.extend(("--resume", str(last_path)))
            print(f"Resuming after epoch {last_epoch}; target is {target_epochs}.")
        run(command)
    else:
        print(f"Training already reached epoch {last_epoch}/{target_epochs}; skipping.")

    if not best_path.is_file():
        raise SystemExit(f"Training did not produce the required checkpoint: {best_path}")

    run(
        [
            sys.executable,
            "-u",
            "scripts/find_best_threshold.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(best_path),
            "--update-config",
        ]
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "src.evaluation.evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(best_path),
            "--no-save-predictions",
        ]
    )

    summary_path = ROOT / "outputs" / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dice") is None:
        raise SystemExit("Evaluation completed without a real Dice value.")

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    archive = package_results(config_path, output)
    print("\nPROJECT COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {archive}")


if __name__ == "__main__":
    main()
