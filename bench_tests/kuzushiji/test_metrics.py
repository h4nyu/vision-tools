import torch
from bench.kuzushiji.metrics import Metrics
from vision_tools import Boxes, Labels, Points


def test_metrics() -> None:
    points = Points(
        torch.tensor(
            [
                [
                    3,
                    3,
                ],
                [
                    24,
                    25,
                ],
            ]
        )
    )
    labels = Labels(
        torch.tensor(
            [
                0,
                1,
            ]
        )
    )
    gt_boxes = Boxes(
        torch.tensor(
            [
                [
                    0,
                    0,
                    11,
                    11,
                ],
                [
                    19,
                    19,
                    31,
                    31,
                ],
            ]
        )
    )
    gt_labels = Labels(
        torch.tensor(
            [
                0,
                1,
            ]
        )
    )

    metrics = Metrics()
    tp, fp, fn = metrics.add(
        points=points, labels=labels, gt_boxes=gt_boxes, gt_labels=gt_labels
    )
    assert tp == 2
    assert fp == 0
    assert fn == 0
    score = metrics()
    assert score == 1.0
