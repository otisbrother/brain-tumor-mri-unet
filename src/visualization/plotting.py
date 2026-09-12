"""Consistent visualization of MRI, masks, and overlays."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Robustly scale non-zero MRI intensities to [0, 1]."""
    values = np.asarray(image, dtype=np.float32)
    output = np.zeros_like(values)
    nonzero = values != 0
    if nonzero.any():
        low, high = np.percentile(values[nonzero], (1, 99))
        if high > low:
            output = np.clip((values - low) / (high - low), 0, 1)
    return output


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.1, 0.1), alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask over a grayscale image."""
    base = normalize_for_display(image)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    foreground = mask.astype(bool)
    rgb[foreground] = (1 - alpha) * rgb[foreground] + alpha * np.asarray(color)
    return np.clip(rgb, 0, 1)


def create_slice_figure(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Create original/mask/overlay panels, optionally including ground truth."""
    cache_dir = Path("outputs/.matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("FLAIR", normalize_for_display(image), "gray"),
        ("Prediction", predicted_mask, "gray"),
        ("Prediction overlay", overlay_mask(image, predicted_mask), None),
    ]
    if ground_truth is not None:
        panels.extend(
            [("Ground truth", ground_truth, "gray"), ("Ground-truth overlay", overlay_mask(image, ground_truth, (0.1, 1.0, 0.1)), None)]
        )
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (label, values, color_map) in zip(np.atleast_1d(axes), panels):
        axis.imshow(values, cmap=color_map)
        axis.set_title(label)
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
    return figure
