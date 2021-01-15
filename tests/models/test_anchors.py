import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import List
from object_detection.entities import (
    PascalBoxes,
    ImageBatch,
)
from object_detection.models.anchors import Anchors
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize(
    "scales, ratios, size",
    [
        (
            [0.5, 0.9],
            [1.0, 0.75],
            2,
        ),
        (
            [0.5, 0.9],
            [1.0, 0.75],
            3,
        ),
    ],
)
def test_anchors(
    scales: List[float],
    ratios: List[float],
    size: int,
) -> None:
    base_size = 1024 + 512
    stride = 2 ** 7
    h = base_size // stride
    w = base_size // stride
    images = ImageBatch(torch.zeros((1, 3, h, w), dtype=torch.float32))
    fn = Anchors(size=size, scales=scales, ratios=ratios)
    res = fn(images, stride)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    assert base_size == res.max()
    assert 0 == res.min()
    plot = DetectionPlot(torch.ones((3, base_size, base_size)))
    plot.draw_boxes(boxes=res[0:num_anchors], color="red")
    plot.draw_boxes(boxes=res[-4:], color="blue")
    plot.save(
        f"store/test-anchors-{size}-{stride}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png"
    )
