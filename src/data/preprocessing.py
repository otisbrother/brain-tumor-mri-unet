"""MRI and segmentation preprocessing operations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

BRATS_LABELS = frozenset({0, 1, 2, 4})


def validate_brats_labels(mask: np.ndarray) -> set[int]:
    """Return labels and raise if a BraTS mask contains unexpected values."""
    if not np.isfinite(mask).all():
        raise ValueError("Segmentation mask contains NaN or Inf values.")
    unique_values = np.unique(mask)
    unexpected_values = unique_values[~np.isin(unique_values, tuple(BRATS_LABELS))]
    if unexpected_values.size:
        raise ValueError(
            f"Unexpected BraTS labels: {unexpected_values.tolist()}; expected only 0, 1, 2, 4."
        )
    labels = {int(value) for value in unique_values}
    return labels


def convert_brats_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert BraTS labels 1, 2, and 4 to binary whole-tumor foreground."""
    validate_brats_labels(mask)
    return (mask > 0).astype(np.float32)


def zscore_normalize_nonzero(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Z-score normalize non-zero brain voxels and preserve zero background."""
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D MRI array, got shape {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("MRI volume contains NaN or Inf values.")
    nonzero = data != 0
    normalized = np.zeros_like(data, dtype=np.float32)
    if not nonzero.any():
        return normalized
    values = data[nonzero]
    mean = float(values.mean())
    std = float(values.std())
    if std <= epsilon:
        normalized[nonzero] = data[nonzero] - mean
    else:
        normalized[nonzero] = (data[nonzero] - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced NaN or Inf values.")
    return normalized


def resize_slice(array: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
    """Resize a 2D slice with bilinear MRI or nearest-neighbor mask interpolation."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {array.shape}.")
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    if array.shape == (target_size, target_size):
        result = np.asarray(array, dtype=np.float32)
        return (result > 0.5).astype(np.float32) if is_mask else result
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    mode = "nearest" if is_mask else "bilinear"
    options = {} if is_mask else {"align_corners": False}
    resized = functional.interpolate(tensor, size=(target_size, target_size), mode=mode, **options)
    result = resized[0, 0].numpy().astype(np.float32, copy=False)
    return (result > 0.5).astype(np.float32) if is_mask else result


def calculate_tumor_area(positive_pixels: int, spacing_xy: tuple[float, float] | None) -> dict[str, float | int | None]:
    """Calculate slice lesion area from pixel count and optional physical spacing."""
    result: dict[str, float | int | None] = {
        "positive_pixels": int(positive_pixels),
        "area_mm2": None,
        "area_cm2": None,
    }
    if spacing_xy is not None:
        sx, sy = (float(value) for value in spacing_xy)
        if not np.isfinite((sx, sy)).all() or min(sx, sy) <= 0:
            raise ValueError(f"Invalid pixel spacing: {spacing_xy}")
        area_mm2 = positive_pixels * sx * sy
        result.update(area_mm2=float(area_mm2), area_cm2=float(area_mm2 / 100.0))
    return result


def calculate_tumor_volume(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float | int]:
    """Calculate lesion volume using positive voxels and NIfTI voxel spacing."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
        raise ValueError(f"Invalid voxel spacing: {spacing}")
    voxels = int(np.count_nonzero(mask))
    volume_mm3 = float(voxels * np.prod(spacing_array))
    return {
        "tumor_voxels": voxels,
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_mm3 / 1000.0,
    }
