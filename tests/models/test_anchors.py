import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import List
from object_detection.entities import (
    Boxes,
    ImageBatch,
)
from object_detection.models.anchors import Anchors
from object_detection.utils import DetectionPlot


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
    images = ImageBatch(torch.zeros((1, 3, h, w), dtype=torch.float32))
    fn = Anchors(size=size, scales=scales, ratios=ratios)
    res = fn(images, stride)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    assert 0 == res.min()
    plot = DetectionPlot(torch.ones((3, original_h, original_w)))
    plot.draw_boxes(boxes=Boxes(res[3:4]), color="red")
    plot.draw_boxes(boxes=Boxes(res[-4:-3]), color="blue")
    plot.save(
        f"store/test-anchors-{size}-{stride}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png"
    )
