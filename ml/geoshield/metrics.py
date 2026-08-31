"""Dataset-independent segmentation metrics."""

from __future__ import annotations

import numpy as np


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, num_classes: int = 5, ignore_index: int = 255) -> np.ndarray:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    precision = true_positive / np.maximum(matrix.sum(axis=0), 1.0)
    recall = true_positive / np.maximum(matrix.sum(axis=1), 1.0)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def iou_from_confusion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - true_positive
    return true_positive / np.maximum(union, 1.0)


def summarize_metrics(matrix: np.ndarray) -> dict[str, object]:
    f1 = f1_from_confusion(matrix)
    iou = iou_from_confusion(matrix)
    damage_f1 = f1[1:]
    return {
        "localization_f1": float((matrix[1:, 1:].trace()) / max(matrix[1:, :].sum() + matrix[:, 1:].sum() - matrix[1:, 1:].trace(), 1)),
        "damage_macro_f1": float(damage_f1.mean()),
        "per_class_f1": [float(value) for value in f1],
        "per_class_iou": [float(value) for value in iou],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
