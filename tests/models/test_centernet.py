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
    ToBoxes,
)
from app.utils import plot_heatmap, plot_boxes
def test_toboxes() -> None:
    keymap = torch.zeros((1, 1, 10, 10))
    keymap[:, :, 3:6, 3:6] = torch.tensor(
        [[0.0, 0.1, 0.0,], [0.1, 0.3, 0.1,], [0.0, 0.1, 0.0,],],
    )

    keymap[:, :, 6:9, 6:9] = torch.tensor(
        [[0.0, 0.5, 0.0,], [0.2, 0.6, 0.3,], [0.0, 0.4, 0.0,],],
    )
    sizemap = torch.zeros((1, 2, 10, 10))
    sizemap[:, :, 4, 4] = torch.tensor([0.3, 0.4]).view(1, 2)
    sizemap[:, :, 7, 7] = torch.tensor([0.1, 0.2]).view(1, 2)
    fn = ToBoxes(thresold=0.1)
    preds = fn(keymap, sizemap)
    for probs, boxes in preds:
        plot_boxes("/store/plot/test-toboxes.png", boxes, probs)


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
