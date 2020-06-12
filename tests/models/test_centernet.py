import typing as t
import numpy as np
import torch
from app.dataset import Targets
from app.models.centernet import (
    #  BoxRegression,
    #  LabelClassification,
    Backbone,
    CenterNet,
    CenterHeatMap,
    Criterion,
    FocalLoss,
)
from app.utils import plot_heatmap


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


def test_heatmap() -> None:
    boxes = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.2],
            [0.2, 0.25, 0.1, 0.1],
            [0.3, 0.4, 0.1, 0.1],
            [0.51, 0.51, 0.1, 0.1],
            [0.5, 0.5, 0.3, 0.1],
        ]
    )
    fn = CenterHeatMap(w=512, h=512)
    res = fn(boxes)[0]
    c, _, _ = res.shape
    plot_heatmap(res[0], f"/store/plot/test-heatmap.png")
    res = res[res > 0.99]
    print(res.shape)
    #  plot_heatmap(res, f"/store/plot/threshold-heatmap.png")

    #  for i in range(c):
    #      plot_heatmap(res[i], f"/store/plot/test-heatmap-{i}.png")


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
