"""Losses used by both GeoShield segmentation models."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def weighted_cross_entropy(logits, target, class_weights=None, ignore_index: int = 255):
    if F is None:
        raise RuntimeError("Install PyTorch before calculating training losses")
    return F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)


def soft_dice_loss(logits, target, class_weights=None, ignore_index: int = 255, smooth: float = 1.0):
    if torch is None or F is None:
        raise RuntimeError("Install PyTorch before calculating training losses")
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_float = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    reduce_dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        return 1.0 - (dice * weights).sum()
    return 1.0 - dice.mean()


def combined_segmentation_loss(logits, target, class_weights=None, ignore_index: int = 255):
    ce = weighted_cross_entropy(logits, target, class_weights, ignore_index)
    dice = soft_dice_loss(logits, target, class_weights, ignore_index)
    return 0.5 * ce + 0.5 * dice, {"cross_entropy": float(ce.detach().cpu()), "soft_dice": float(dice.detach().cpu())}

