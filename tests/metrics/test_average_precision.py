import numpy as np
import torch
from object_detection import Labels, Boxes, Confidences
from object_detection.metrics.average_precision import AveragePrecision, auc


def test_average_precision() -> None:
    metrics = AveragePrecision(iou_threshold=0.3)
    boxes = Boxes(
        torch.tensor(
            [
                [15, 15, 25, 25],
                [0, 0, 15, 15],
                [25, 25, 35, 35],
            ]
        )
    )
    confidences = Confidences(torch.tensor([0.9, 0.8, 0.7]))

    gt_boxes = Boxes(
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
    assert round(res, 4) == round((1 / 2 * 1 / 2 + 0.0 * 1 / 2), 4)


def test_no_gt_has_box() -> None:
    metrics = AveragePrecision(iou_threshold=0.3)
    boxes = Boxes(
        torch.tensor(
            [
                [15, 15, 25, 25],
                [0, 0, 15, 15],
                [25, 25, 35, 35],
            ]
        )
    )
    confidences = Confidences(torch.tensor([0.9, 0.8, 0.7]))

    gt_boxes = Boxes(torch.tensor([]))
    metrics.add(
        boxes,
        confidences,
        gt_boxes,
    )

    res = metrics()
    assert round(res, 4) == round(0, 4)


def test_no_gt_and_box() -> None:
    metrics = AveragePrecision(iou_threshold=0.3)
    boxes = Boxes(torch.tensor([]))
    confidences = Confidences(torch.tensor([]))

    gt_boxes = Boxes(torch.tensor([]))
    metrics.add(
        boxes,
        confidences,
        gt_boxes,
    )

    res = metrics()
    assert round(res, 4) == round(1.0, 4)


def test_has_gt_no_box() -> None:
    metrics = AveragePrecision(iou_threshold=0.3)
    boxes = Boxes(torch.tensor([]))
    confidences = Confidences(torch.tensor([]))

    gt_boxes = Boxes(
        torch.tensor(
            [
                [15, 15, 25, 25],
            ]
        )
    )
    metrics.add(
        boxes,
        confidences,
        gt_boxes,
    )

    res = metrics()
    assert round(res, 4) == round(0, 4)
