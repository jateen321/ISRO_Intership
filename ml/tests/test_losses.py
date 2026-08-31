import pytest


torch = pytest.importorskip("torch")

from geoshield.losses import combined_segmentation_loss, soft_dice_loss


def test_loss_ignores_unclassified_pixels_and_is_finite():
    logits = torch.zeros(2, 5, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :2, :2] = 255
    loss, details = combined_segmentation_loss(logits, target, torch.ones(5))
    assert torch.isfinite(loss)
    assert set(details) == {"cross_entropy", "soft_dice"}
    loss.backward()
    assert logits.grad is not None


def test_all_ignored_dice_is_a_valid_zero_result():
    logits = torch.zeros(1, 5, 4, 4)
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    assert torch.isfinite(soft_dice_loss(logits, target))

