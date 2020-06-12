import typing as t
import numpy as np
import torch
from app.dataset import Targets
from app.models.centernet import (
    #  BoxRegression,
    #  LabelClassification,
    Backbone,
    CenterNet,
    Criterion,
    FocalLoss,
    HardHeatMap,
    SoftHeatMap,
    ToPosition,
)
from app.utils import plot_heatmap


def test_topos() -> None:
    heatmap = torch.tensor(
        [
            [
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.6, 0.7, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.6, 0.7, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                ],
            ],
        ]
    )
    print(heatmap.shape)
    fn = ToPosition(thresold=0.1)
    pos = fn(heatmap)
    print(pos)


def test_focal_loss() -> None:
    heatmaps = torch.tensor([[1, 1, 0.0], [1, 1, 1], [1, 1, 1],])
    fn = FocalLoss()
    preds = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1, 1.0], [1.0, 1.0, 1.0],])
    res = fn(preds, heatmaps)
    print(res)

    preds = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    res = fn(preds, heatmaps)
    print(res)

    preds = torch.tensor([[0.0, 0.0, 1], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    res = fn(preds, heatmaps)
    print(res)


def test_hardheatmap() -> None:
    boxes = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.2],
            [0.2, 0.25, 0.1, 0.1],
            [0.3, 0.4, 0.1, 0.1],
            [0.51, 0.51, 0.1, 0.1],
            [0.5, 0.5, 0.3, 0.1],
        ]
    )
    fn = HardHeatMap(w=512, h=512)
    res = fn(boxes)
    plot_heatmap(res[0][0], f"/store/plot/test-hard-heatmap.png")
    pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    res = (pool(res) == res) & (res > 0.5)
    plot_heatmap(res[0][0], f"/store/plot/test-hard-pooled.png")


def test_softheatmap() -> None:
    boxes = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.2],
            [0.2, 0.25, 0.1, 0.1],
            [0.3, 0.4, 0.1, 0.1],
            [0.51, 0.51, 0.1, 0.1],
            [0.5, 0.5, 0.3, 0.1],
        ]
    )
    fn = SoftHeatMap(w=512, h=512)
    res = fn(boxes)
    plot_heatmap(res[0][0], f"/store/plot/test-soft-heatmap.png")
    pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    res = (pool(res) == res) & (res > 0.5)
    plot_heatmap(res[0][0], f"/store/plot/test-soft-pooled.png")


def test_backbone() -> None:
    inputs = torch.rand((10, 3, 1024, 1024))
    fn = Backbone("resnet34", out_channels=128)
    outs = fn(inputs)
    for o in outs:
        assert o.shape[1] == 128


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    num_classes = 1
    fn = CenterNet(num_classes=num_classes)
    outc, outr = fn(inputs)
    assert outc.shape == (1, num_classes, 1024 / 2, 1024 / 2)
