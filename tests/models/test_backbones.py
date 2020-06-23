import typing as t
import numpy as np
import torch
from object_detection.models.backbones import (
    ResNetBackbone,
    EfficientNetBackbone,
)


def test_resnetbackbone() -> None:
    inputs = torch.rand((10, 3, 1024, 1024))
    fn = ResNetBackbone("resnet34", out_channels=128)
    outs = fn(inputs)
    for o in outs:
        assert o.shape[1] == 128


def test_effnetbackbone() -> None:
    inputs = torch.rand((10, 3, 512, 512))
    fn = EfficientNetBackbone(1, out_channels=128)
    outs = fn(inputs)
    for o in outs:
        assert o.shape[1] == 128
