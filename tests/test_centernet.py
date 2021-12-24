import typing as t
import pytest
import numpy as np
import torch
from typing import Any
from torch import Tensor
from vision_tools.centernet import (
    CenterNet,
    ToBoxes,
    HMLoss,
    ToPoints,
)
from vision_tools.box import yolo_to_coco
from vision_tools.backbones.resnet import (
    ResNetBackbone,
)
from torch.utils.data import DataLoader


def test_hm_loss() -> None:
    heatmaps = torch.tensor(
        [
            [0.1, 0.5, 0.1],
            [0.5, 1, 0.5],
            [0.1, 0.5, 0.1],
        ]
    )
    fn = HMLoss(beta=4)
    preds = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    res = fn(preds, heatmaps)
    assert res < 1e-4

    preds = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 1, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    res = fn(preds, heatmaps)
    assert res < 0.03

    preds = torch.tensor(
        [
            [0.5, 0.0, 0.5],
            [0.0, 1, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    res = fn(preds, heatmaps)
    assert res < 0.3


def test_centernet_forward() -> None:
    inputs = torch.rand((1, 3, 512, 512))
    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    fn = CenterNet(num_classes=2, channels=channels, backbone=backbone, out_idx=3)

    netout = fn(inputs)
    heatmap, sizemap, _ = netout
    assert heatmap.shape == (1, 2, 512 // 2, 512 // 2)
    assert sizemap.shape == (1, 4, 512 // 2, 512 // 2)


def test_to_boxes() -> None:
    h, w = (3, 3)
    heatmap: Any = torch.rand((1, 2, h, w))
    sizemap: Any = torch.rand((1, 4, h, w))
    anchors: Any = torch.rand((4, h, w))
    channels = 32
    to_boxes = ToBoxes()
    netout = (heatmap, sizemap, anchors)
    box_batch, conf_batch, label_batch = to_boxes(netout)
    assert len(box_batch) == len(conf_batch) == len(label_batch)


def test_to_points() -> None:
    h, w = (3, 3)
    heatmaps: Any = torch.rand((1, 2, h, w))
    to_points = ToPoints()
    point_batch, conf_batch, label_batch = to_points(heatmaps, h, w)
    assert len(point_batch) == len(conf_batch) == len(label_batch)
