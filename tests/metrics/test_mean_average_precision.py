import numpy as np
import torch
from vnet import Labels, Boxes, Confidences
from vnet.metrics.mean_average_precision import MeanAveragePrecision


def test_half() -> None:
    metrics = MeanAveragePrecision(num_classes=2, iou_threshold=0.3)
    boxes = Boxes(
        torch.tensor(
            [
                [15, 15, 25, 25],
                [0, 0, 15, 15],
                [25, 25, 35, 35],
            ]
        )
    )

    confidences = Confidences(
        torch.tensor(
            [
                0.9,
                0.8,
                0.7,
            ]
        )
    )

    labels = Labels(
        torch.tensor(
            [
                0,
                0,
                1,
            ]
        )
    )

    gt_boxes = Boxes(
        torch.tensor(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
            ]
        )
    )

    gt_labels = Labels(torch.tensor([0, 0]))

    metrics.add(
        boxes,
        confidences,
        labels,
        gt_boxes,
        gt_labels,
    )
    score, scores = metrics()
    print(score)
    print(scores)
