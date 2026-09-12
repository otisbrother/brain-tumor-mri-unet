"""NIfTI input/output with metadata preservation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - gives a focused runtime message
    nib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class NiftiData:
    """Loaded NIfTI voxel data and spatial metadata."""

    array: np.ndarray
    affine: np.ndarray
    header: Any
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError("nibabel is required. Install dependencies with: pip install -r requirements.txt")


def load_nifti(path: str | Path, dtype: np.dtype = np.float32) -> NiftiData:
    """Load a finite 3D NIfTI volume without changing orientation."""
    _require_nibabel()
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    try:
        image = nib.load(str(file_path))
        array = np.asarray(image.dataobj, dtype=dtype)
    except Exception as exc:
        raise ValueError(f"Could not read NIfTI file {file_path}: {exc}") from exc
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"NIfTI volume is empty: {file_path}")
    if not np.isfinite(array).all():
        raise ValueError(f"NIfTI contains NaN or Inf values: {file_path}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError(f"Invalid voxel spacing {spacing} in {file_path}")
    orientation = tuple(str(code) for code in nib.aff2axcodes(image.affine))
    return NiftiData(array, image.affine.copy(), image.header.copy(), spacing, orientation)


def save_nifti(
    array: np.ndarray,
    affine: np.ndarray,
    header: Any,
    path: str | Path,
    dtype: np.dtype = np.uint8,
) -> Path:
    """Save a 3D array using spatial metadata copied from its source MRI."""
    _require_nibabel()
    output_path = Path(path)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D output array, got shape {array.shape}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copied_header = header.copy()
    copied_header.set_data_dtype(dtype)
    image = nib.Nifti1Image(array.astype(dtype, copy=False), affine, copied_header)
    nib.save(image, str(output_path))
    return output_path

