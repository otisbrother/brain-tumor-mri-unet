"""Lazy patient-safe datasets for 2D slice training and 3D inference."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from .nifti import NiftiData, load_nifti
from .preprocessing import convert_brats_mask_to_binary, resize_slice, zscore_normalize_nonzero

SampleTransform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def read_split_ids(path: str | Path) -> list[str]:
    """Read unique patient IDs from a split file."""
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    ids = [line.strip().lower() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate patient IDs in split file: {split_path}")
    return ids


def load_metadata_subset(metadata_csv: str | Path, patient_ids: list[str]) -> pd.DataFrame:
    """Load metadata rows in the exact order specified by a patient split."""
    frame = pd.read_csv(metadata_csv, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].str.lower()
    indexed = frame.set_index("patient_id", drop=False)
    missing = [patient_id for patient_id in patient_ids if patient_id not in indexed.index]
    if missing:
        raise ValueError(f"Split contains patient IDs absent from metadata: {missing[:10]}")
    return indexed.loc[patient_ids].reset_index(drop=True)


class BraTSSliceDataset(Dataset[dict[str, Any]]):
    """Lazy axial-slice dataset created only after patient-level splitting."""

    def __init__(
        self,
        metadata_csv: str | Path,
        split_file: str | Path,
        modalities: list[str],
        target_size: int,
        split_name: str,
        keep_negative_slices: bool = True,
        negative_to_positive_ratio: float = 1.0,
        transform: SampleTransform | None = None,
        seed: int = 42,
        cache_size: int = 2,
        max_patients: int | None = None,
        index_cache_dir: str | Path | None = None,
    ) -> None:
        if not modalities:
            raise ValueError("At least one modality is required.")
        if target_size <= 0:
            raise ValueError("target_size must be positive.")
        self.modalities = [item.lower() for item in modalities]
        self.target_size = int(target_size)
        self.split_name = split_name.lower()
        self.transform = transform
        self.cache_size = max(0, int(cache_size))
        self.index_cache_dir = Path(index_cache_dir) if index_cache_dir else None
        patient_ids = read_split_ids(split_file)
        if max_patients is not None:
            if max_patients <= 0:
                raise ValueError("max_patients must be positive when provided.")
            patient_ids = patient_ids[:max_patients]
        self.metadata = load_metadata_subset(metadata_csv, patient_ids)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.samples = self._build_slice_index(
            keep_negative_slices=keep_negative_slices,
            negative_to_positive_ratio=float(negative_to_positive_ratio),
            seed=seed,
        )

    def _build_slice_index(
        self, keep_negative_slices: bool, negative_to_positive_ratio: float, seed: int
    ) -> list[tuple[int, int]]:
        positive: list[tuple[int, int]] = []
        negative: list[tuple[int, int]] = []
        for row_index, row in tqdm(self.metadata.iterrows(), total=len(self.metadata),
                                   desc=f"Index {self.split_name}", mininterval=2):
            cache_path = None
            if self.index_cache_dir is not None:
                sources = [Path(row[f"{name}_path"]) for name in ["seg", *self.modalities]]
                signature = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                             for path in sources]
                key = hashlib.sha256(json.dumps([1, signature]).encode()).hexdigest()
                self.index_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = self.index_cache_dir / f"{key}.json"
                if cache_path.is_file():
                    try:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        pos, neg = cached["positive"], cached["negative"]
                        if not all(type(z) is int and z >= 0 for z in pos + neg):
                            raise ValueError("Invalid cached slice indices")
                        positive.extend((row_index, z) for z in pos)
                        negative.extend((row_index, z) for z in neg)
                        continue
                    except (ValueError, KeyError, TypeError):
                        pass  # Rebuild interrupted or malformed cache files.
            mask = convert_brats_mask_to_binary(load_nifti(row["seg_path"]).array)
            brain = np.zeros(mask.shape, dtype=bool)
            for modality in self.modalities:
                column = f"{modality}_path"
                if column not in row or not isinstance(row[column], str) or not row[column]:
                    raise ValueError(f"{row['patient_id']}: missing metadata path for {modality}.")
                volume = load_nifti(row[column]).array
                if volume.shape != mask.shape:
                    raise ValueError(f"{row['patient_id']}: {modality} and SEG shapes differ.")
                brain |= volume != 0
            tumor_by_slice = mask.any(axis=(0, 1))
            brain_by_slice = brain.any(axis=(0, 1))
            pos = [int(z) for z in np.flatnonzero(tumor_by_slice)]
            neg = [int(z) for z in np.flatnonzero(brain_by_slice & ~tumor_by_slice)]
            positive.extend((row_index, z) for z in pos)
            negative.extend((row_index, z) for z in neg)
            if cache_path is not None:
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"positive": pos, "negative": neg}), encoding="utf-8")
                temporary.replace(cache_path)
        if not keep_negative_slices:
            self.positive_sample_keys = set(positive)
            return positive
        if self.split_name == "train" and positive and negative_to_positive_ratio >= 0:
            requested = min(len(negative), int(round(len(positive) * negative_to_positive_ratio)))
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(negative), size=requested, replace=False) if requested else []
            negative = [negative[int(index)] for index in sorted(chosen)]
        self.positive_sample_keys = set(positive)
        # Patient-grouped order avoids reloading an entire NIfTI volume for
        # nearly every slice. Training randomness is supplied by the sampler.
        samples = sorted(positive + negative)
        if not samples:
            raise ValueError(f"No usable brain slices found for split {self.split_name!r}.")
        return samples

    def _load_patient(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.metadata.iloc[row_index]
        patient_id = str(row["patient_id"])
        if patient_id in self._cache:
            self._cache.move_to_end(patient_id)
            return self._cache[patient_id]
        channels = [zscore_normalize_nonzero(load_nifti(row[f"{modality}_path"]).array) for modality in self.modalities]
        image = np.stack(channels, axis=0).astype(np.float32, copy=False)
        mask = convert_brats_mask_to_binary(load_nifti(row["seg_path"]).array)
        if self.cache_size:
            self._cache[patient_id] = (image, mask)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return image, mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, slice_index = self.samples[index]
        row = self.metadata.iloc[row_index]
        volume, mask_volume = self._load_patient(row_index)
        channels = [resize_slice(volume[channel, :, :, slice_index], self.target_size) for channel in range(volume.shape[0])]
        image = torch.from_numpy(np.stack(channels, axis=0)).float()
        mask = torch.from_numpy(resize_slice(mask_volume[:, :, slice_index], self.target_size, is_mask=True))[None].float()
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "patient_id": str(row["patient_id"]),
            "slice_index": int(slice_index),
        }


class PatientGroupedSampler(Sampler[int]):
    """Shuffle patients and their slices while keeping each patient contiguous."""

    def __init__(self, dataset: BraTSSliceDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[int, list[int]] = {}
        for sample_index, (row_index, _) in enumerate(dataset.samples):
            self.groups.setdefault(row_index, []).append(sample_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        patient_order = list(self.groups)
        rng.shuffle(patient_order)
        for row_index in patient_order:
            sample_indices = self.groups[row_index].copy()
            rng.shuffle(sample_indices)
            yield from sample_indices


class BraTSVolumeDataset(Dataset[dict[str, Any]]):
    """Patient-level volume dataset for evaluation and optional paired inference."""

    def __init__(self, metadata_csv: str | Path, split_file: str | Path, modalities: list[str]) -> None:
        self.modalities = [item.lower() for item in modalities]
        self.metadata = load_metadata_subset(metadata_csv, read_split_ids(split_file))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        nifti_volumes: list[NiftiData] = [load_nifti(row[f"{modality}_path"]) for modality in self.modalities]
        shape = nifti_volumes[0].array.shape
        if any(item.array.shape != shape for item in nifti_volumes):
            raise ValueError(f"{row['patient_id']}: modality shapes do not match.")
        image = np.stack([zscore_normalize_nonzero(item.array) for item in nifti_volumes], axis=0)
        mask_data = load_nifti(row["seg_path"])
        if mask_data.array.shape != shape:
            raise ValueError(f"{row['patient_id']}: MRI and segmentation shapes do not match.")
        return {
            "image": image.astype(np.float32, copy=False),
            "mask": convert_brats_mask_to_binary(mask_data.array),
            "patient_id": str(row["patient_id"]),
            "affine": nifti_volumes[0].affine,
            "header": nifti_volumes[0].header,
            "spacing": nifti_volumes[0].spacing,
        }
