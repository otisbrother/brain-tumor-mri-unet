import pytest
import torch

from src.training.losses import BCEDiceLoss, DiceLoss


def test_dice_loss_prefers_identical_confident_mask() -> None:
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    correct_logits = torch.where(target.bool(), torch.tensor(20.0), torch.tensor(-20.0))
    wrong_logits = -correct_logits
    loss = DiceLoss(smooth=1.0)
    assert loss(correct_logits, target).item() == pytest.approx(0.0, abs=1e-6)
    assert loss(wrong_logits, target) > 0.5


def test_combined_loss_has_finite_backward_pass() -> None:
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    value = BCEDiceLoss(0.5, 0.5, 1.0)(logits, target)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

