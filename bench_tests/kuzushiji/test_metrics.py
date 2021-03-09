import torch
from bench.kuzushiji.metrics import Metrics
from vnet import Boxes, Labels


def test_metrics() -> None:
    boxes = Boxes(
        torch.tensor(
            [
                [
                    1,
                    1,
                    10,
                    10,
                ],
                [
                    20,
                    20,
                    30,
                    30,
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
        boxes=boxes, labels=labels, gt_boxes=gt_boxes, gt_labels=gt_labels
    )
    assert tp == 2
    assert fp == 0
    assert fn == 0
    score = metrics()
    assert score == 1.0
