import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import List
from object_detection.entities import (
    PyramidIdx,
    PascalBoxes,
    ImageBatch,
)
from object_detection.models.anchors import Anchors
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize(
    "stride, size, scales, ratios",
    [
        (
            64,
            2,
            [1.0],
            [1.0, 2.0, 0.5],
        ),
    ],
)
def test_anchors(
    stride: int,
    size: float,
    scales: List[float],
    ratios: List[float],
) -> None:
    base_size = 1024
    h = base_size // stride
    w = base_size // stride
    images = ImageBatch(
        torch.zeros((1, 3, h, w), dtype=torch.float32)
    )
    fn = Anchors(size=size, scales=scales, ratios=ratios)
    res = fn(images, stride)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    offset = num_anchors * h * (w // 2) + num_anchors * h // 2
    ids = [offset + x for x in range(num_anchors)]
    plot = DetectionPlot(w=base_size, h=base_size)
    plot.with_pascal_boxes(PascalBoxes(res[ids]), color="red")
    plot.save(
        f"store/test-anchors-{stride}-{size}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png"
    )


