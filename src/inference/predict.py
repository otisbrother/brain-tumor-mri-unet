"""Run batched 2D U-Net inference across a complete NIfTI volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from src.data.nifti import NiftiData, load_nifti, save_nifti
from src.data.preprocessing import calculate_tumor_area, calculate_tumor_volume, zscore_normalize_nonzero
from src.inference.postprocessing import postprocess_volume
from src.models.unet import UNet
from src.utils.config import ensure_output_directories, load_config
from src.utils.device import get_device


def build_model(config: dict[str, Any]) -> UNet:
    """Build the configured U-Net and verify supported V1 settings."""
    model_config = config["model"]
    if model_config.get("name", "unet").lower() != "unet":
        raise ValueError("V1 supports only model.name=unet.")
    return UNet(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 32)),
        bilinear=bool(model_config.get("bilinear", True)),
    )


def load_model_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a structured checkpoint and return an evaluation-mode model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    device = device or get_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (RuntimeError, OSError, ValueError) as exc:
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing model_state_dict.")
    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config).to(device)
    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    if (
        bool(checkpoint_config.get("training", {}).get("multi_gpu", False))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        model = torch.nn.DataParallel(model)
    model.eval()
    return model, checkpoint


def predict_volume_probabilities(
    model: torch.nn.Module,
    volume: np.ndarray,
    target_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return full-resolution HxWxD probability maps from CxHxWxD MRI input."""
    if volume.ndim != 4:
        raise ValueError(f"Expected CxHxWxD volume, got shape {volume.shape}.")
    if volume.shape[0] <= 0 or batch_size <= 0 or target_size <= 0:
        raise ValueError("Channels, batch_size, and target_size must be positive.")
    if not np.isfinite(volume).all():
        raise ValueError("MRI input contains NaN or Inf.")
    normalized = np.stack([zscore_normalize_nonzero(channel) for channel in volume], axis=0)
    slices = torch.from_numpy(np.moveaxis(normalized, -1, 0)).float()
    original_size = tuple(int(value) for value in volume.shape[1:3])
    model.eval()
    current_batch_size = min(batch_size, len(slices))
    while True:
        probabilities: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(slices), current_batch_size):
                    batch = slices[start : start + current_batch_size].to(device)
                    resized = functional.interpolate(
                        batch, size=(target_size, target_size), mode="bilinear", align_corners=False
                    )
                    logits = model(resized)
                    if logits.shape[1] != 1:
                        raise ValueError(f"Binary inference expects one output channel, got {logits.shape[1]}.")
                    batch_probabilities = torch.sigmoid(logits)
                    if batch_probabilities.shape[-2:] != original_size:
                        batch_probabilities = functional.interpolate(
                            batch_probabilities, size=original_size, mode="bilinear", align_corners=False
                        )
                    probabilities.append(batch_probabilities[:, 0].cpu())
            break
        except torch.cuda.OutOfMemoryError as exc:
            if device.type != "cuda" or current_batch_size <= 1:
                raise RuntimeError(
                    "CUDA ran out of memory even at inference batch size 1. Reduce data.target_size."
                ) from exc
            current_batch_size = max(1, current_batch_size // 2)
            probabilities.clear()
            torch.cuda.empty_cache()
    depth_first = torch.cat(probabilities, dim=0).numpy()
    return np.moveaxis(depth_first, 0, -1).astype(np.float32, copy=False)


def predict_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    config: dict[str, Any],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Predict, postprocess, and summarize a complete MRI volume."""
    device = device or next(model.parameters()).device
    probabilities = predict_volume_probabilities(
        model,
        volume,
        target_size=int(config["data"]["target_size"]),
        batch_size=int(config["inference"]["batch_size"]),
        device=device,
    )
    threshold = float(config["inference"]["threshold"])
    raw_mask = (probabilities >= threshold).astype(np.uint8)
    post_config = config.get("postprocessing", {})
    mask = (
        postprocess_volume(raw_mask, int(post_config.get("min_component_pixels", 20)))
        if post_config.get("enabled", True)
        else raw_mask
    )
    volume_stats = calculate_tumor_volume(mask, spacing)
    areas = mask.sum(axis=(0, 1))
    if int(areas.max(initial=0)) > 0:
        important_slice = int(np.argmax(areas))
    else:
        brain_slices = np.flatnonzero(np.any(volume != 0, axis=(0, 1, 2)))
        important_slice = int(brain_slices[len(brain_slices) // 2]) if brain_slices.size else volume.shape[-1] // 2
    lesion_probabilities = probabilities[mask.astype(bool)]
    area = calculate_tumor_area(int(areas[important_slice]), spacing[:2])
    minimum_pixels = int(config["inference"].get("minimum_positive_pixels", 1))
    return {
        "probabilities": probabilities,
        "raw_mask": raw_mask,
        "mask": mask,
        "important_slice": important_slice,
        "detected": int(mask.sum()) >= minimum_pixels,
        "mean_lesion_probability": float(lesion_probabilities.mean()) if lesion_probabilities.size else None,
        "important_slice_area": area,
        **volume_stats,
    }


def _patient_id(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True, help="MRI path; repeat in configured modality order.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    device = get_device()
    model, checkpoint = load_model_checkpoint(config, args.checkpoint, device)
    model_config = checkpoint.get("config", config)
    modalities = model_config["data"]["modalities"]
    if len(args.input) != len(modalities):
        raise SystemExit("Provide one --input for each checkpoint modality, in its configured order.")
    loaded: list[NiftiData] = [load_nifti(path) for path in args.input]
    if any(item.array.shape != loaded[0].array.shape for item in loaded):
        raise SystemExit("Input modality shapes do not match.")
    result = predict_volume(model, np.stack([item.array for item in loaded]), loaded[0].spacing, config, device)
    output_dir = args.output_dir or Path(config["paths"]["output_dir"]) / "predictions"
    patient_id = _patient_id(args.input[0])
    mask_path = save_nifti(result["mask"], loaded[0].affine, loaded[0].header, output_dir / f"{patient_id}_pred_seg.nii.gz")
    summary = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    summary["mask_path"] = str(mask_path)
    summary_path = output_dir / f"{patient_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
