#!/usr/bin/env python
"""Validate BraTS file presence, geometry, finite values, and segmentation labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.discovery import discover_patient_directories, inspect_patient  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--allow-missing-optional-modalities", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/evaluation/dataset_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    directories = discover_patient_directories(config["data"]["root_dir"])
    if not directories:
        raise SystemExit(f"No patient folders found under {config['data']['root_dir']}")
    valid: list[str] = []
    invalid: list[dict[str, str]] = []
    for directory in directories:
        try:
            row = inspect_patient(directory, require_all_modalities=not args.allow_missing_optional_modalities)
            valid.append(str(row["patient_id"]))
            print(f"OK   {row['patient_id']} shape={row['height']}x{row['width']}x{row['depth']} labels={row['segmentation_labels']}")
        except (FileNotFoundError, ValueError) as exc:
            invalid.append({"directory": str(directory), "error": str(exc)})
            print(f"FAIL {directory.name}: {exc}")
    report = {"total": len(directories), "valid": len(valid), "invalid": len(invalid), "invalid_cases": invalid}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

