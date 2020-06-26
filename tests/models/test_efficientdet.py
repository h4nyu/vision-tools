import torch
from object_detection.entities.image import ImageBatch
from object_detection.models.efficientdet import (
    ClipBoxes,
    BBoxTransform,
    RegressionModel,
    ClassificationModel,
    EfficientDet,
)
from object_detection.models.backbones import EfficientNetBackbone


def test_clip_boxes() -> None:
    images = torch.ones((1, 1, 10, 10))
    boxes = torch.tensor([[[14, 0, 20, 0]]])
    fn = ClipBoxes()
    res = fn(boxes, images)

    assert (
        res - torch.tensor([[[14, 0, 10, 0]]])
    ).sum() == 0  # TODO ??? [10, 0, 10, 0]


def test_bbox_transform() -> None:
    boxes = torch.tensor([[[2, 2, 20, 6], [4, 2, 8, 6],]])

    deltas = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1],]])
    fn = BBoxTransform()
    res = fn(boxes, deltas)


def test_regression_model() -> None:
    images = torch.ones((1, 1, 10, 10))
    fn = RegressionModel(1, num_anchors=9)
    res = fn(images)
    assert res.shape == (1, 900, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(num_features_in=100, num_classes=2)
    res = fn(images)
    assert res.shape == (1, 900, 2)


def test_effdet() -> None:
    images = ImageBatch(torch.ones((1, 3, 512, 512)))
    annotations = torch.ones((1, 10, 5))
    channels = 32
    backbone = EfficientNetBackbone(1, out_channels=channels, pretrained=True)
    fn = EfficientDet(num_classes=2, backbone=backbone, channels=32,)
    res = fn(images)
    print(res)
