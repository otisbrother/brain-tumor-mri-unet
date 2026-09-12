import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from src.data.nifti import load_nifti, save_nifti


def test_nifti_round_trip_preserves_shape_affine_and_spacing(tmp_path) -> None:
    source_path = tmp_path / "source.nii.gz"
    affine = np.diag([0.5, 0.75, 2.0, 1.0])
    source = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine)
    nib.save(source, source_path)
    loaded = load_nifti(source_path)
    output_path = save_nifti(np.zeros((4, 5, 6), dtype=np.uint8), loaded.affine, loaded.header, tmp_path / "mask.nii.gz")
    output = load_nifti(output_path)
    assert output.array.shape == (4, 5, 6)
    np.testing.assert_allclose(output.affine, affine)
    assert output.spacing == pytest.approx((0.5, 0.75, 2.0))

