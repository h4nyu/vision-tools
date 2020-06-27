import torch
import pytest
from typing import Any
from object_detection.entities.image import ImageBatch
from object_detection.entities.box import YoloBoxes, Labels
from object_detection.models.efficientdet import (
    ClipBoxes,
    RegressionModel,
    ClassificationModel,
    EfficientDet,
    Criterion,
    LabelLoss,
    BoxLoss,
)
from object_detection.models.anchors import Anchors
from object_detection.models.backbones import EfficientNetBackbone


def test_clip_boxes() -> None:
    images = torch.ones((1, 1, 10, 10))
    boxes = torch.tensor([[[14, 0, 20, 0]]])
    fn = ClipBoxes()
    res = fn(boxes, images)

    assert (
        res - torch.tensor([[[14, 0, 10, 0]]])
    ).sum() == 0  # TODO ??? [10, 0, 10, 0]


def test_regression_model() -> None:
    c, h, w = 4, 10, 10
    num_anchors = 9
    images = torch.ones((1, c, h, w))
    fn = RegressionModel(in_channels=c, num_anchors=num_anchors)
    res = fn(images)
    assert res.shape == (1, h * w * num_anchors, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(num_features_in=100, num_classes=2)
    res = fn(images)
    assert res.shape == (1, 900, 2)


@pytest.mark.parametrize(
    "preds,expected",
    [
        ([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0],], 1e-4),
        ([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0],], 1e-4),
        ([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0],], 1e1),
        ([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0],], 2e1),
    ],
)
def test_label_loss(preds: Any, expected: float) -> None:
    iou_max = torch.tensor([0.0, 0.4, 0.6,])
    match_indices = torch.tensor([0, 1, 0,])
    pred_classes = torch.tensor(preds)
    gt_classes = torch.tensor([0, 1])
    fn = LabelLoss()
    res = fn(
        iou_max=iou_max,
        match_indices=match_indices,
        pred_classes=pred_classes,
        gt_classes=gt_classes,
    )
    assert res < expected


@pytest.mark.parametrize(
    "preds,expected",
    [
        ([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0],], 1e-4),
        ([[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0],], 1e-4),
        ([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.5, 0.0, 0.0, 0.0],], 1e0),
    ],
)
def test_box_loss(preds: Any, expected: float) -> None:
    iou_max = torch.tensor([0.0, 0.4, 0.6,])
    match_indices = torch.tensor([0, 1, 0,])
    anchors = torch.tensor(
        [[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1],]
    )
    pred_boxes = torch.tensor(preds)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1],])
    fn = BoxLoss()
    res = fn(
        iou_max=iou_max,
        match_indices=match_indices,
        anchors=anchors,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
    )
    assert res < expected


def test_effdet() -> None:
    images = ImageBatch(torch.ones((1, 3, 512, 512)))
    annotations = torch.ones((1, 10, 5))
    channels = 32
    backbone = EfficientNetBackbone(1, out_channels=channels, pretrained=True)
    fn = EfficientDet(num_classes=2, backbone=backbone, channels=32,)
    anchors, boxes, labels = fn(images)
    assert anchors.shape == boxes.shape[1:]
    assert labels.shape[:2] == boxes.shape[:2]
