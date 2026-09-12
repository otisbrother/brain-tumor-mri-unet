import pytest
import torch

from src.evaluation.metrics import binary_segmentation_metrics


def test_identical_masks_have_perfect_metrics() -> None:
    mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    metrics = binary_segmentation_metrics(mask, mask)
    for name in ("dice", "iou", "precision", "recall", "specificity"):
        assert metrics[name] == pytest.approx(1.0)


def test_non_overlap_has_zero_dice_and_iou() -> None:
    prediction = torch.tensor([[1, 0], [0, 0]], dtype=torch.float32)
    target = torch.tensor([[0, 1], [0, 0]], dtype=torch.float32)
    metrics = binary_segmentation_metrics(prediction, target)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0


def test_empty_mask_conventions_are_finite() -> None:
    empty = torch.zeros(2, 2)
    both_empty = binary_segmentation_metrics(empty, empty)
    false_positive = binary_segmentation_metrics(torch.ones(2, 2), empty)
    assert both_empty["dice"] == 1.0
    assert false_positive["dice"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in both_empty.values())

