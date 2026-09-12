import numpy as np
import pytest

from src.data.preprocessing import convert_brats_mask_to_binary


def test_brats_labels_convert_to_whole_tumor() -> None:
    mask = np.array([[0, 1, 2, 4]], dtype=np.int16)
    converted = convert_brats_mask_to_binary(mask)
    np.testing.assert_array_equal(converted, np.array([[0, 1, 1, 1]], dtype=np.float32))


def test_invalid_label_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="Unexpected BraTS labels"):
        convert_brats_mask_to_binary(np.array([0, 3], dtype=np.int16))
    with pytest.raises(ValueError, match="Unexpected BraTS labels"):
        convert_brats_mask_to_binary(np.array([0.0, 0.5], dtype=np.float32))
