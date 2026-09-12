from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.dataset import BraTSSliceDataset, BraTSVolumeDataset, PatientGroupedSampler
from src.data.discovery import build_metadata, discover_patient_directories


def _write_patient(root: Path, patient_id: str, tumor_slice: int) -> None:
    directory = root / patient_id
    directory.mkdir()
    affine = np.diag([0.5, 0.75, 2.0, 1.0])
    image = np.zeros((12, 10, 5), dtype=np.float32)
    image[2:10, 2:8, 1:4] = 1.0
    mask = np.zeros_like(image, dtype=np.int16)
    mask[4:7, 4:7, tumor_slice] = 4
    for modality in ("flair", "t1", "t1ce", "t2"):
        nib.save(nib.Nifti1Image(image, affine), directory / f"{patient_id}_{modality}.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), directory / f"{patient_id}_seg.nii.gz")


def test_metadata_and_lazy_datasets_keep_patient_boundaries(tmp_path) -> None:
    dataset_root = tmp_path / "BraTS2021_Training_Data"
    dataset_root.mkdir()
    for index in range(3):
        _write_patient(dataset_root, f"BraTS2021_{index:05d}", tumor_slice=index + 1)
    metadata_path = tmp_path / "metadata.csv"
    metadata = build_metadata(dataset_root, metadata_path)
    assert len(metadata) == 3
    assert metadata["tumor_voxel_count"].tolist() == [9, 9, 9]
    assert metadata["tumor_volume_mm3"].tolist() == [9 * 0.5 * 0.75 * 2.0] * 3

    train_path = tmp_path / "train.txt"
    test_path = tmp_path / "test.txt"
    train_path.write_text("brats2021_00000\nbrats2021_00001\n", encoding="utf-8")
    test_path.write_text("brats2021_00002\n", encoding="utf-8")
    slices = BraTSSliceDataset(metadata_path, train_path, ["flair"], 16, "train", seed=42)
    item = slices[0]
    assert item["image"].shape == (1, 16, 16)
    assert item["mask"].shape == (1, 16, 16)
    assert item["patient_id"] in {"brats2021_00000", "brats2021_00001"}
    volumes = BraTSVolumeDataset(metadata_path, test_path, ["flair"])
    assert volumes[0]["patient_id"] == "brats2021_00002"
    assert volumes[0]["image"].shape == (1, 12, 10, 5)


def test_discovery_ignores_loose_files_from_multiple_subjects(tmp_path) -> None:
    _write_patient(tmp_path, "BraTS2021_00001", tumor_slice=1)
    _write_patient(tmp_path, "BraTS2021_00002", tumor_slice=2)
    # Simulate duplicate patient files placed at the extraction root.
    for patient_id in ("BraTS2021_00001", "BraTS2021_00002"):
        source = tmp_path / patient_id / f"{patient_id}_flair.nii.gz"
        (tmp_path / source.name).write_bytes(source.read_bytes())
    discovered = discover_patient_directories(tmp_path)
    assert {path.name for path in discovered} == {"BraTS2021_00001", "BraTS2021_00002"}


def test_patient_limit_and_grouped_sampler(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for index in range(3):
        _write_patient(dataset_root, f"BraTS2021_{index:05d}", tumor_slice=index + 1)
    metadata_path = tmp_path / "metadata.csv"
    build_metadata(dataset_root, metadata_path)
    split_path = tmp_path / "train.txt"
    split_path.write_text(
        "brats2021_00000\nbrats2021_00001\nbrats2021_00002\n", encoding="utf-8"
    )
    limited = BraTSSliceDataset(
        metadata_path, split_path, ["flair"], 16, "train", max_patients=1
    )
    assert len(limited.metadata) == 1

    complete = BraTSSliceDataset(metadata_path, split_path, ["flair"], 16, "train")
    order = list(PatientGroupedSampler(complete, seed=42))
    row_order = [complete.samples[index][0] for index in order]
    transitions = sum(left != right for left, right in zip(row_order, row_order[1:]))
    assert transitions == len(complete.metadata) - 1
