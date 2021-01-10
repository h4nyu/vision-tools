import numpy as np
import torch
from object_detection.entities import Labels, PascalBoxes, Confidences
from object_detection.metrics.average_precision import AveragePrecision, auc


def test_average_precision() -> None:
    metrics = AveragePrecision(iou_threshold=0.3)
    boxes = PascalBoxes(
        torch.tensor(
            [
                [15, 15, 25, 25],
                [0, 0, 15, 15],
                [25, 25, 35, 35],
            ]
        )
    )
    confidences = Confidences(torch.tensor([0.9, 0.8, 0.7]))

    gt_boxes = PascalBoxes(
        torch.tensor(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
            ]
        )
    )
    metrics.add(
        boxes,
        confidences,
        gt_boxes,
    )

    res = metrics()
    assert round(res, 4) == round((1 / 2 * 1 / 2 + 1 / 3 * 1 / 2), 4)
