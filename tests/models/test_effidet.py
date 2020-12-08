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
)
from object_detection.models.anchors import Anchors
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)


def test_regression_model() -> None:
    c, h, w = 4, 10, 10
    num_anchors = 9
    images = torch.ones((1, c, h, w))
    fn = RegressionModel(in_channels=c, num_anchors=num_anchors, out_size=c)
    res = fn(images)
    assert res.shape == (1, h * w * num_anchors, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(in_channels=100, num_classes=2)
    res = fn(images)
    assert res.shape == (1, 900, 2)


def test_effdet() -> None:
    images = ImageBatch(torch.ones((1, 3, 512, 512)))
    annotations = torch.ones((1, 10, 5))
    channels = 32
    backbone = EfficientNetBackbone(1, out_channels=channels, pretrained=True)
    fn = EfficientDet(
        num_classes=2,
        backbone=backbone,
        channels=32,
    )
    anchors, boxes, labels = fn(images)
    for x, y in zip(labels, boxes):
        assert x.shape[:2] == y.shape[:2]
