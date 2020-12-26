import torch
import pytest
from typing import Any
from object_detection.entities.image import ImageBatch
from object_detection.entities.box import PascalBoxes, Labels
from object_detection.models.effidet import (
    RegressionModel,
    ClassificationModel,
    EfficientDet,
    Criterion,
    BoxDiff,
    ToBoxes,
)
from object_detection.models.anchors import Anchors
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
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


def test_effdet_forward() -> None:
    images = ImageBatch(torch.ones((1, 3, 512, 512)))
    annotations = torch.ones((1, 10, 5))
    channels = 32
    to_boxes = ToBoxes()
    backbone = EfficientNetBackbone(
        1, out_channels=channels, pretrained=True
    )
    net = EfficientDet(
        num_classes=2,
        backbone=backbone,
        channels=32,
    )
    netout = net(images)
    anchors, boxes, labels = netout

    # for x, y in zip(labels, boxes):
    #     assert x.shape[:2] == y.shape[:2]
    to_boxes(netout)

