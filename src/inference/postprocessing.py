"""Conservative connected-component filtering for binary slice masks."""

from __future__ import annotations

from collections import deque

import numpy as np


def remove_small_components_2d(mask: np.ndarray, min_component_pixels: int = 20) -> np.ndarray:
    """Remove 8-connected foreground components smaller than a pixel threshold."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}.")
    if min_component_pixels <= 1:
        return (mask > 0).astype(np.uint8)
    foreground = mask.astype(bool)
    output = np.zeros_like(foreground)
    visited = np.zeros_like(foreground)
    height, width = foreground.shape
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for row, col in np.argwhere(foreground):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        queue = deque([(row, col)])
        visited[row, col] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_col = queue.popleft()
            component.append((current_row, current_col))
            for delta_row, delta_col in neighbors:
                next_row, next_col = current_row + delta_row, current_col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and foreground[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        if len(component) >= min_component_pixels:
            rows, cols = zip(*component)
            output[rows, cols] = True
    return output.astype(np.uint8)


def postprocess_volume(mask: np.ndarray, min_component_pixels: int = 20) -> np.ndarray:
    """Apply conservative component filtering independently to axial slices."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got {mask.shape}.")
    result = np.zeros(mask.shape, dtype=np.uint8)
    for slice_index in range(mask.shape[2]):
        result[:, :, slice_index] = remove_small_components_2d(mask[:, :, slice_index], min_component_pixels)
    return result

