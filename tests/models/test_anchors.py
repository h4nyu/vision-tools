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
    "scales, ratios",
    [
        (
            [0.5, 0.9],
            [1.0, 0.75],
        ),
    ],
)
def test_anchors(
    scales: List[float],
    ratios: List[float],
) -> None:
    base_size = 1024
    size = 1
    stride=512
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
    print(res)
    plot = DetectionPlot(w=base_size, h=base_size)
    plot.with_pascal_boxes(res, color="red")
    plot.save(
        f"store/test-anchors-{stride}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png"
    )


