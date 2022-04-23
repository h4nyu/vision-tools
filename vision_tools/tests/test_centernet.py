import typing as t
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from vision_tools.centernet import CenterNet, CenterNetHead, HMLoss, ToBoxes, ToPoints


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


def test_head() -> None:
    in_channels = [
        32,
        64,
        128,
    ]
    hidden_channels = 48
    num_classes = 3
    head = CenterNetHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
    )
    feats = [torch.rand(2, c, 32, 32) for c in in_channels]
    res = head(feats)
    assert len(feats) == len(res)
    for f, r in zip(feats, res):
        assert r.size() == (2, 4 + 1 + num_classes, r.size(2), r.size(3))


# def test_centernet_forward() -> None:
#     inputs = torch.rand((1, 3, 512, 512))
#     channels = 32
#     backbone = ResNetBackbone("resnet34", out_channels=channels)
#     fn = CenterNet(num_classes=2, channels=channels, backbone=backbone, out_idx=3)

#     netout = fn(inputs)
#     heatmap, sizemap, _ = netout
#     assert heatmap.shape == (1, 2, 512 // 2, 512 // 2)
#     assert sizemap.shape == (1, 4, 512 // 2, 512 // 2)


# def test_to_boxes() -> None:
#     h, w = (3, 3)
#     heatmap: Any = torch.rand((1, 2, h, w))
#     sizemap: Any = torch.rand((1, 4, h, w))
#     anchors: Any = torch.rand((4, h, w))
#     channels = 32
#     to_boxes = ToBoxes()
#     netout = (heatmap, sizemap, anchors)
#     box_batch, conf_batch, label_batch = to_boxes(netout)
#     assert len(box_batch) == len(conf_batch) == len(label_batch)


# def test_to_points() -> None:
#     h, w = (3, 3)
#     heatmaps: Any = torch.rand((1, 2, h, w))
#     to_points = ToPoints()
#     point_batch, conf_batch, label_batch = to_points(heatmaps, h, w)
#     assert len(point_batch) == len(conf_batch) == len(label_batch)
