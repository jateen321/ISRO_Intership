import numpy as np

from geoshield.metrics import confusion_matrix, f1_from_confusion, iou_from_confusion


def test_metrics_ignore_unclassified_pixels():
    target = np.array([[0, 1, 2, 255]])
    prediction = np.array([[0, 1, 1, 4]])
    matrix = confusion_matrix(target, prediction, num_classes=5)
    assert matrix.sum() == 3
    assert np.isclose(f1_from_confusion(matrix)[1], 2 / 3)
    assert iou_from_confusion(matrix)[2] == 0.0
