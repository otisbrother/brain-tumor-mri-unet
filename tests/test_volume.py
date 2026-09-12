import numpy as np
import pytest

from src.data.preprocessing import calculate_tumor_area, calculate_tumor_volume


def test_volume_uses_voxel_spacing() -> None:
    mask = np.zeros((3, 4, 5), dtype=np.uint8)
    mask.ravel()[:10] = 1
    result = calculate_tumor_volume(mask, (0.5, 0.5, 2.0))
    assert result["tumor_voxels"] == 10
    assert result["volume_mm3"] == pytest.approx(5.0)
    assert result["volume_cm3"] == pytest.approx(0.005)


def test_area_without_spacing_reports_only_pixels() -> None:
    result = calculate_tumor_area(20, None)
    assert result == {"positive_pixels": 20, "area_mm2": None, "area_cm2": None}

