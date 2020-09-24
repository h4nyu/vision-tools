import torch
import pytest
import typing as t
import torch.nn.functional as F
from object_detection.models.losses import (
    DIoU,
    GIoU,
    DIoULoss,
    IoULoss,
    FocalLoss,
)
from object_detection import PascalBoxes
from torch import nn, Tensor


def test_iouloss() -> None:
    fn = IoULoss()
    pred_boxes = PascalBoxes(
        torch.tensor([[0.2, 0.2, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3]])
    )
    tgt_boxes = PascalBoxes(
        torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]])
    )
    iouloss, union = fn(
        pred_boxes,
        tgt_boxes,
    )
    assert (iouloss - 0.6250) < 1e-7


def test_diouloss() -> None:
    fn = DIoULoss()
    pred_boxes = PascalBoxes(
        torch.tensor(
            [
                [0.2, 0.2, 0.3, 0.3],
                [0.2, 0.2, 0.3, 0.3],
            ]
        )
    )
    tgt_boxes = PascalBoxes(
        torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]])
    )
    res = fn(
        pred_boxes,
        tgt_boxes,
    )
    assert (res - 0).abs() < 1e-7


def test_giou() -> None:
    fn = GIoU()
    pred_boxes = PascalBoxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.2, 0.2],
                [0.2, 0.2, 0.3, 0.3],
            ]
        )
    )
    tgt_boxes = PascalBoxes(torch.tensor([[0.2, 0.2, 0.3, 0.3]]))
    res = fn(
        pred_boxes,
        tgt_boxes,
    )
    assert res.shape == (2, 1)
    assert F.l1_loss(res, torch.tensor([[1.5], [0.0]])) < 1e-7


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            [
                [-10, 10],
                [10, -10],
            ],
            1e-6,
        ),
        (
            [
                [10, 10],
                [10, -10],
            ],
            0.35,
        ),
        (
            [
                [10, 10],
                [10, 10],
            ],
            0.70,
        ),
        (
            [
                [10, -10],
                [10, -10],
            ],
            19,
        ),
    ],
)
def test_focal_loss(values: t.Any, expected: float) -> None:
    fn = FocalLoss()
    pred = torch.Tensor(values).softmax(1)

    target = torch.Tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    res = fn(pred, target).sum()
    assert res < expected
