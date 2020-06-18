import typing as t
import numpy as np
import torch
from app.models.centernet import (
    CenterNet,
    Criterion,
    FocalLoss,
    HardHeatMap,
    SoftHeatMap,
    ToBoxes,
    TTAMerge,
    NetOutputs,
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
    preds = fn(NetOutputs(heatmap=keymap, sizemap=sizemap))
    plot = DetectionPlot()
    plot.with_image(keymap[0, 0])
    for pred in preds:
        plot.with_boxes(pred.boxes, pred.confidences)
    plot.with_boxes(gt_boxes, color="blue")
    plot.save("/store/plot/test-toboxes.png",)


def test_focal_loss() -> None:
    heatmaps = torch.tensor([[0.0, 0.5, 0.0], [0.5, 1, 0.5], [0, 0.5, 0],])
    fn = FocalLoss()
    preds = torch.tensor(
        [[0.0001, 0.5, 0.0001], [0.5, 0.999, 0.5], [0.001, 0.5, 0.0001],]
    )
    res = fn(preds, heatmaps)
    print(res)
    #
    #  preds = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    #  res = fn(preds, heatmaps)
    #  print(res)
    #
    #  preds = torch.tensor([[0.0, 0.0, 1], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    #  res = fn(preds, heatmaps)
    #  print(res)


def test_heatmap_box_conversion() -> None:
    in_boxes = torch.tensor([[0.2, 0.4, 0.1, 0.3]])
    to_boxes = ToBoxes(thresold=0.1)
    to_heatmap = HardHeatMap(w=64, h=64)
    res = to_heatmap(in_boxes)
    hm = res.heatmap
    sm = res.sizemap
    assert sm.shape == (1, 2, 64, 64)
    assert hm.shape == (1, 1, 64, 64)
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[25, 12]])).sum() == 0
    assert (sm[0, :, 25, 12] - torch.tensor([0.1, 0.3])).sum() == 0
    out_boxes = next(iter(to_boxes(res))).boxes
    assert out_boxes[0, 0] < in_boxes[0, 0]
    assert out_boxes[0, 1] < in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot()
    plot.with_image(hm[0, 0])
    plot.with_boxes(in_boxes, color="blue")
    plot.with_boxes(out_boxes)
    plot.save(f"/store/plot/test-hard-heatmap.png")


def test_softheatmap() -> None:
    in_boxes = torch.tensor([[0.2, 0.4, 0.1, 0.3]])
    to_boxes = ToBoxes(thresold=0.1)
    to_heatmap = SoftHeatMap(w=64, h=64)
    res = to_heatmap(in_boxes)
    hm = res.heatmap
    sm = res.sizemap
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[25, 12]])).sum() == 0
    assert (sm.nonzero()[0, 2:] - torch.tensor([[25, 12]])).sum() == 0
    assert hm.shape == (1, 1, 64, 64)
    assert sm.shape == (1, 2, 64, 64)
    assert (sm[0, :, 25, 12] - torch.tensor([0.1, 0.3])).sum() == 0
    out_boxes = next(iter(to_boxes(res))).boxes
    assert out_boxes[0, 0] < in_boxes[0, 0]
    assert out_boxes[0, 1] < in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot()
    plot.with_image(hm[0, 0])
    plot.with_boxes(in_boxes, color="blue")
    plot.with_boxes(out_boxes)
    plot.save(f"/store/plot/test-soft-heatmap.png")


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    fn = CenterNet()
    out = fn(inputs)
    assert out.heatmap.shape == (1, 1, 1024 // 2, 1024 // 2)
