"""Numerically safe binary segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _as_binary_numpy(values: ArrayLike, threshold: float, from_logits: bool) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().float().cpu()
        if from_logits:
            tensor = torch.sigmoid(tensor)
        array = tensor.numpy()
    else:
        array = np.asarray(values, dtype=np.float32)
        if from_logits:
            array = 1.0 / (1.0 + np.exp(-np.clip(array, -80, 80)))
    if not np.isfinite(array).all():
        raise ValueError("Metric input contains NaN or Inf.")
    return array >= threshold


def confusion_counts(
    prediction: ArrayLike, target: ArrayLike, threshold: float = 0.5, from_logits: bool = False
) -> dict[str, int]:
    """Count TP, FP, FN, and TN after thresholding prediction and target."""
    predicted = _as_binary_numpy(prediction, threshold, from_logits)
    expected = _as_binary_numpy(target, 0.5, False)
    if predicted.shape != expected.shape:
        raise ValueError(f"Prediction and target shapes differ: {predicted.shape} vs {expected.shape}")
    return {
        "tp": int(np.count_nonzero(predicted & expected)),
        "fp": int(np.count_nonzero(predicted & ~expected)),
        "fn": int(np.count_nonzero(~predicted & expected)),
        "tn": int(np.count_nonzero(~predicted & ~expected)),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute metrics, defining a perfect score when both relevant sets are empty."""
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))

    def safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 1.0

    return {
        "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "sensitivity": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
    }


def binary_segmentation_metrics(
    prediction: ArrayLike, target: ArrayLike, threshold: float = 0.5, from_logits: bool = False
) -> dict[str, Any]:
    """Return confusion counts and derived overlap/classification metrics.

    Convention: a metric with a zero denominator is 1.0. Thus an empty
    prediction paired with an empty ground truth receives Dice/IoU 1.0, while
    a false-positive prediction against empty ground truth receives 0.0.
    """
    counts = confusion_counts(prediction, target, threshold=threshold, from_logits=from_logits)
    return {**counts, **metrics_from_counts(counts)}

