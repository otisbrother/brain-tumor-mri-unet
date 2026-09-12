"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its essential sections."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    required = {"project", "data", "model", "training", "inference", "paths"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    modalities = config["data"].get("modalities", [])
    if not modalities:
        raise ValueError("data.modalities must contain at least one MRI modality.")
    configured_channels = int(config["model"].get("in_channels", len(modalities)))
    if configured_channels != len(modalities):
        raise ValueError(
            "model.in_channels must equal len(data.modalities): "
            f"{configured_channels} != {len(modalities)}"
        )
    return config


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create model and generated-output directories."""
    model_dir = Path(config["paths"]["model_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "figures", "predictions", "evaluation"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

