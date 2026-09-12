import numpy as np
import pytest

from src.data.preprocessing import resize_slice, zscore_normalize_nonzero


def test_nonzero_zscore_preserves_background_and_is_finite() -> None:
    volume = np.zeros((4, 4, 2), dtype=np.float32)
    volume[1:3, 1:3, :] = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    normalized = zscore_normalize_nonzero(volume)
    foreground = volume != 0
    assert normalized.dtype == np.float32
    assert normalized.shape == volume.shape
    assert np.isfinite(normalized).all()
    assert np.all(normalized[~foreground] == 0)
    assert normalized[foreground].mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized[foreground].std() == pytest.approx(1.0, abs=1e-6)


def test_constant_and_empty_volumes_do_not_create_nan() -> None:
    constant = np.ones((3, 3, 3), dtype=np.float32) * 7
    assert np.isfinite(zscore_normalize_nonzero(constant)).all()
    assert np.all(zscore_normalize_nonzero(np.zeros_like(constant)) == 0)


def test_mask_resize_remains_binary() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.float32)
    resized = resize_slice(mask, 9, is_mask=True)
    assert resized.shape == (9, 9)
    assert set(np.unique(resized)) <= {0.0, 1.0}


def test_resize_skips_interpolation_when_shape_already_matches() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    resized = resize_slice(image, 4)
    np.testing.assert_array_equal(resized, image)
    assert np.shares_memory(resized, image)
