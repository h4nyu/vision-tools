import torch
import pytest
import typing as t
import torch.nn.functional as F
from vision_tools.losses import (
    DIoU,
    GIoU,
    DIoULoss,
    IoULoss,
    FocalLoss,
    SigmoidFocalLoss,
)
from vision_tools import Boxes
from torch import nn, Tensor


def test_iouloss() -> None:
    fn = IoULoss()
    pred_boxes = Boxes(torch.tensor([[0.2, 0.2, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3]]))
    tgt_boxes = Boxes(torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]]))
    iouloss, union = fn(
        pred_boxes,
        tgt_boxes,
    )
    assert (iouloss - 0.6250) < 1e-7


def test_diouloss() -> None:
    fn = DIoULoss()
    pred_boxes = Boxes(
        torch.tensor(
            [
                [0.2, 0.2, 0.3, 0.3],
                [0.2, 0.2, 0.3, 0.3],
            ]
        )
    )
    tgt_boxes = Boxes(torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]]))
    res = fn(
        pred_boxes,
        tgt_boxes,
    )
    assert (res - 0).abs() < 1e-7


def test_giou() -> None:
    fn = GIoU()
    pred_boxes = Boxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.2, 0.2],
                [0.2, 0.2, 0.3, 0.3],
            ]
        )
    )
    tgt_boxes = Boxes(torch.tensor([[0.2, 0.2, 0.3, 0.3]]))
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
            7,
        ),
        (
            [
                [10, 10],
                [10, 10],
            ],
            14,
        ),
    ],
)
def test_focal_loss(values: t.Any, expected: float) -> None:
    fn = FocalLoss()
    pred = torch.Tensor(values).sigmoid()

    target = torch.Tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    res = fn(pred, target).sum()
    assert res < expected


def test_focal_loss_empty() -> None:
    fn = FocalLoss()
    pred = torch.Tensor([[-10, -10]]).sigmoid()

    target = torch.Tensor([])
    res = fn(pred, target).sum()
    assert res < 1e-6


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            [-10, 10],
            1e-6,
        ),
        (
            [10, 10],
            0.35,
        ),
    ],
)
def test_sigmoid_focal_loss(values: t.Any, expected: float) -> None:
    fn = SigmoidFocalLoss()
    source = torch.tensor([[values]]).float()
    target = torch.tensor(
        [
            [
                0,
                1,
            ]
        ]
    )
    res = fn(source, target).sum()
