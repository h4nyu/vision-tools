from typing import Any

import pytest
import torch

from vision_tools.anchors import Anchors
from vision_tools.effidet import (
    ClassificationModel,
    Criterion,
    EfficientDet,
    RegressionModel,
    ToBoxes,
)


def test_regression_model() -> None:
    n, c, h, w = 2, 4, 10, 10
    num_anchors = 9
    images = torch.ones((n, c, h, w))
    fn = RegressionModel(in_channels=c, num_anchors=num_anchors)
    res = fn(images)
    assert res.shape == (n, h * w * num_anchors, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(in_channels=100, num_classes=2, depth=1)
    res = fn(images)
    assert res.shape == (1, 900, 2)


# def test_effdet_to_box() -> None:
#     images = torch.ones((1, 3, 512, 512))
#     annotations = torch.ones((1, 10, 5))
#     channels = 32
#     to_boxes = ToBoxes()
#     backbone = EfficientNetBackbone(1, out_channels=channels, pretrained=True)
#     net = EfficientDet(
#         num_classes=2,
#         backbone=backbone,
#         channels=32,
#     )
#     netout = net(images)
#     for boxes, confidences, labels in zip(*to_boxes(netout)):
#         assert len(boxes) == len(confidences) == len(labels)
