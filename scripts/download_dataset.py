#!/usr/bin/env python
"""Download BraTS 2021 Task 1 through the authenticated Kaggle CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DATASET_ID = "dschettler8845/brats-2021-task1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--unzip", action="store_true", help="Ask Kaggle to unzip its outer download archive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI was not found. Install requirements and configure Kaggle authentication first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", DATASET_ID, "-p", str(args.output_dir)]
    if args.unzip:
        command.append("--unzip")
    subprocess.run(command, check=True)
    print(f"Downloaded {DATASET_ID} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

