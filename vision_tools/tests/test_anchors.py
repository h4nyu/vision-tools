from typing import *
import torch, pytest, numpy as np
from pathlib import Path
from vision_tools.anchors import Anchors


@pytest.mark.parametrize(
    "scales, ratios, size",
    [
        (
            [1.0],
            [1.0],
            1,
        ),
        (
            [2 / 3],
            [0.75],
            2,
        ),
        (
            [3 / 2],
            [1.25],
            3,
        ),
    ],
)
def test_anchors(
    scales: List[float],
    ratios: List[float],
    size: int,
) -> None:
    original_w = 1024 + 512
    original_h = 1024
    stride = 2 ** 7
    h = original_h // stride
    w = original_w // stride
    images = torch.zeros((1, 3, h, w), dtype=torch.float32)
    fn = Anchors(size=size, scales=scales, ratios=ratios)
    res = fn(images, stride)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    # assert 0 == res.min() // TODO