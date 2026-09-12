import numpy as np

from src.inference.postprocessing import remove_small_components_2d


def test_small_components_are_removed_conservatively() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1
    mask[3:6, 3:6] = 1
    cleaned = remove_small_components_2d(mask, min_component_pixels=4)
    assert cleaned[0, 0] == 0
    assert cleaned[3:6, 3:6].sum() == 9

