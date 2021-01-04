import torch
from object_detection.entities.box import PascalBoxes, Confidences
from object_detection.metrics.mean_average_precision import (
    MeanAveragePrecision,
    AveragePrecision,
)


def test_mean_average_precision() -> None:
    metrics = MeanAveragePrecision()


def test_average_precision() -> None:
    boxes = PascalBoxes(
        torch.tensor(
            [
                [2, 2, 4, 4],
                [2, 2, 4, 4],
                [2, 2, 4, 4],
                [3, 3, 5, 5],
            ]
        )
    )
    confidences = Confidences(
        torch.tensor(
            [
                0.8,
                0.8,
                0.9,
            ],
        )
    )
    gt_boxes = PascalBoxes(
        torch.tensor(
            [
                [2, 2, 4, 4],
            ]
        )
    )
    metrics = AveragePrecision(iou_threshold=0.5)
    res = metrics(
        boxes=boxes,
        confidences=confidences,
        gt_boxes=gt_boxes,
    )
    print(res)
    # assert res == 0.5
