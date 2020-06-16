import typing as t
import numpy as np
import torch
from app.dataset import Targets
from app.models.centernet import (
    CenterNet,
    Criterion,
    FocalLoss,
    HardHeatMap,
    SoftHeatMap,
    ToBoxes,
)
from app.utils import plot_heatmap, DetectionPlot


def test_toboxes() -> None:
    w, h = 200, 100
    keymap = torch.zeros((2, 1, h, w))
    keymap[0, :, -1, 0] = 1
    keymap[0, :, 40:43, 60:63] = torch.tensor(
        [[0.0, 0.1, 0.0,], [0.1, 0.3, 0.1,], [0.0, 0.1, 0.0,],],
    )

    keymap[:, :, 60:63, 20:23] = torch.tensor(
        [[0.0, 0.5, 0.0,], [0.2, 0.6, 0.3,], [0.0, 0.4, 0.0,],],
    )
    sizemap = torch.zeros((2, 2, h, w))
    sizemap[:, :, 41, 61] = torch.tensor([0.2, 0.1]).view(1, 2)
    sizemap[:, :, 61, 21] = torch.tensor([0.1, 0.2]).view(1, 2)
    gt_boxes = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.2],
            #  [0.41, 0.61, 0.2, 0.1],
            #  [0.61, 0.21, 0.2, 0.1]
        ]
    )
    fn = ToBoxes(thresold=0.1)
    preds = fn(dict(heatmap=keymap, sizemap=sizemap))
    plot = DetectionPlot()
    plot.with_image(keymap[0, 0])
    for probs, boxes in preds:
        plot.with_boxes(boxes, probs)
    plot.with_boxes(gt_boxes, color="blue")
    plot.save("/store/plot/test-toboxes.png",)


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


def test_heatmap_box_conversion() -> None:
    in_boxes = torch.tensor([[0.2, 0.4, 0.1, 0.3]])
    to_boxes = ToBoxes(thresold=0.1)
    to_heatmap = HardHeatMap(w=64, h=64)
    heatmap = to_heatmap(in_boxes)
    _, out_boxes = next(iter(to_boxes(heatmap)))
    assert out_boxes[0, 0] < in_boxes[0, 0]
    assert out_boxes[0, 1] < in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]


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


def test_evaluate() -> None:
    ...


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    fn = CenterNet()
    out = fn(dict(images=inputs))
    assert out["heatmap"].shape == (1, 1, 1024 // 2, 1024 // 2)
