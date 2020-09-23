import torch
import pytest
from typing import Any
from object_detection.entities.image import ImageBatch
from object_detection.entities.box import YoloBoxes, Labels
from object_detection.models.effidet import (
    RegressionModel,
    ClassificationModel,
    EfficientDet,
    Criterion,
    LabelLoss,
    BoxDiff,
    BoxLoss,
)
from object_detection.models.anchors import Anchors
from object_detection.models.backbones.effnet import EfficientNetBackbone


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


@pytest.mark.parametrize(
    "preds,expected",
    [
        (
            [
                [0.01],
                [0.01],
                [0.99],
            ],
            0.13,
        ),
        (
            [
                [0.9],
                [0.0],
                [0.9],
            ],
            7.7,
        ),
        # ([[-10.0], [0.0], [-10.0],], 7.7),
    ],
)
def test_label_loss(preds: Any, expected: float) -> None:
    match_score = torch.tensor(
        [
            0.0,
            0.5,
            0.6,
        ]
    )  # [neg, ignore, pos]
    match_indices = torch.tensor(
        [
            0,
            0,
            0,
        ]
    )
    pred_classes = torch.tensor(preds)
    gt_classes = torch.tensor([0, 0])
    fn = LabelLoss(iou_thresholds=(0.45, 0.55))
    res = fn(
        match_score=match_score,
        match_indices=match_indices,
        logits=pred_classes,
        gt_classes=gt_classes,
    )
    print(res)
    # assert res < expected


@pytest.mark.parametrize(
    "preds,expected",
    [
        (
            [
                [
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                ],
                [0.1, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            0.06,
        ),
        (
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.1, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            0.04,
        ),
        (
            [
                [
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                ],
                [0.1, 0.1, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
            ],
            1e0,
        ),
    ],
)
def test_box_loss(preds: Any, expected: float) -> None:
    match_score = torch.tensor(
        [
            0.0,
            0.4,
            0.6,
        ]
    )
    match_indices = torch.tensor(
        [
            0,
            1,
            0,
        ]
    )
    anchors = torch.tensor(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
        ]
    )
    box_diff = BoxDiff(torch.tensor(preds))
    gt_boxes = YoloBoxes(
        torch.tensor(
            [
                [0.5, 0.5, 0.1, 0.1],
                [0.8, 0.8, 0.1, 0.1],
            ]
        )
    )
    fn = BoxLoss()
    res = fn(
        match_score=match_score,
        match_indices=match_indices,
        anchors=anchors,
        box_diff=box_diff,
        gt_boxes=gt_boxes,
    )
    assert res < expected


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
