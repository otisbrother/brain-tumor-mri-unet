"""BraTS patient discovery, validation, and metadata generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .nifti import load_nifti
from .preprocessing import validate_brats_labels

LOGGER = logging.getLogger(__name__)
MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")


def _nifti_files(directory: Path) -> Iterable[Path]:
    yield from directory.glob("*.nii")
    yield from directory.glob("*.nii.gz")


def find_modality_file(patient_dir: str | Path, modality: str) -> Path | None:
    """Find one exact BraTS modality suffix inside a patient directory."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
    suffixes = (f"_{modality}.nii", f"_{modality}.nii.gz")
    matches = [path for path in _nifti_files(Path(patient_dir)) if path.name.lower().endswith(suffixes)]
    if len(matches) > 1:
        raise ValueError(f"Multiple {modality} files found in {patient_dir}: {matches}")
    return matches[0].resolve() if matches else None


def discover_patient_directories(root_dir: str | Path) -> list[Path]:
    """Recursively find directories containing at least one BraTS modality."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    candidates: set[Path] = set()
    for pattern in ("*_seg.nii", "*_seg.nii.gz", "*_flair.nii", "*_flair.nii.gz"):
        candidates.update(path.parent.resolve() for path in root.rglob(pattern))

    directories: list[Path] = []
    for directory in sorted(candidates):
        subject_ids: set[str] = set()
        for path in _nifti_files(directory):
            lower_name = path.name.lower()
            for modality in MODALITIES:
                marker = f"_{modality}.nii"
                if marker in lower_name:
                    subject_ids.add(lower_name.split(marker)[0])
                    break
        if len(subject_ids) == 1:
            directories.append(directory)
        elif len(subject_ids) > 1:
            LOGGER.warning(
                "Ignoring mixed directory %s containing files from %d subjects.",
                directory,
                len(subject_ids),
            )
    return directories


def patient_id_from_directory(patient_dir: str | Path) -> str:
    """Infer a stable patient ID from the directory or a BraTS filename."""
    directory = Path(patient_dir)
    for modality in ("seg", "flair"):
        file_path = find_modality_file(directory, modality)
        if file_path:
            marker = f"_{modality}.nii"
            return file_path.name.lower().split(marker)[0]
    return directory.name.lower()


def inspect_patient(patient_dir: str | Path, require_all_modalities: bool = True) -> dict[str, object]:
    """Validate one patient and return metadata suitable for a CSV row."""
    directory = Path(patient_dir)
    patient_id = patient_id_from_directory(directory)
    paths = {modality: find_modality_file(directory, modality) for modality in MODALITIES}
    required = MODALITIES if require_all_modalities else ("flair", "seg")
    missing = [modality for modality in required if paths[modality] is None]
    if missing:
        raise ValueError(f"{patient_id}: missing required files: {', '.join(missing)}")

    flair = load_nifti(paths["flair"])  # type: ignore[arg-type]
    segmentation = load_nifti(paths["seg"])  # type: ignore[arg-type]
    if flair.array.shape != segmentation.array.shape:
        raise ValueError(
            f"{patient_id}: FLAIR shape {flair.array.shape} does not match SEG {segmentation.array.shape}."
        )
    labels = validate_brats_labels(segmentation.array)
    for modality in ("t1", "t1ce", "t2"):
        if paths[modality] is not None:
            volume = load_nifti(paths[modality])
            if volume.array.shape != flair.array.shape:
                raise ValueError(
                    f"{patient_id}: {modality} shape {volume.array.shape} does not match FLAIR {flair.array.shape}."
                )

    tumor = segmentation.array > 0
    brain_slices = (flair.array != 0).any(axis=(0, 1))
    tumor_slices = tumor.any(axis=(0, 1))
    tumor_voxels = int(tumor.sum())
    voxel_volume = float(np.prod(flair.spacing))
    height, width, depth = flair.array.shape
    row: dict[str, object] = {
        "patient_id": patient_id,
        **{f"{modality}_path": str(paths[modality]) if paths[modality] else "" for modality in MODALITIES},
        "height": height,
        "width": width,
        "depth": depth,
        "voxel_spacing_x": flair.spacing[0],
        "voxel_spacing_y": flair.spacing[1],
        "voxel_spacing_z": flair.spacing[2],
        "orientation": "".join(flair.orientation),
        "segmentation_labels": ",".join(str(label) for label in sorted(labels)),
        "tumor_voxel_count": tumor_voxels,
        "tumor_volume_mm3": tumor_voxels * voxel_volume,
        "positive_slice_count": int(np.count_nonzero(tumor_slices)),
        "negative_brain_slice_count": int(np.count_nonzero(brain_slices & ~tumor_slices)),
    }
    return row


def build_metadata(
    root_dir: str | Path,
    output_csv: str | Path,
    require_all_modalities: bool = True,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    """Inspect all discovered subjects and write deterministic metadata."""
    directories = discover_patient_directories(root_dir)
    if not directories:
        raise ValueError(f"No BraTS patient directories found below {root_dir}")
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for directory in directories:
        try:
            rows.append(inspect_patient(directory, require_all_modalities=require_all_modalities))
        except (FileNotFoundError, ValueError) as exc:
            if not skip_invalid:
                raise
            errors.append(str(exc))
            LOGGER.warning("Skipping invalid patient: %s", exc)
    if not rows:
        raise ValueError("No valid BraTS patients were found.")
    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    if frame["patient_id"].duplicated().any():
        duplicates = frame.loc[frame["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicate patient IDs found: {duplicates}")
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    frame.attrs["validation_errors"] = errors
    return frame
