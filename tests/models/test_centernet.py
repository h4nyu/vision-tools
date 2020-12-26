import typing as t
import pytest
import numpy as np
import torch
from typing import Any
from object_detection.models.centernet import (
    CenterNet,
    ToBoxes,
    HMLoss,
    Trainer,
    Visualize,
    collate_fn,
)
from object_detection.entities import (
    YoloBoxes,
    Image,
    ImageSize,
    ImageBatch,
)
from object_detection.entities.box import yolo_to_coco
from object_detection.utils import DetectionPlot
from object_detection.models.backbones.resnet import (
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
    fn = CenterNet(channels=channels, backbone=backbone, out_idx=3)
    heatmap, sizemap, _ = fn(inputs)
    assert heatmap.shape == (1, 1, 512 // 2, 512 // 2)
