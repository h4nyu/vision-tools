import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import List
from object_detection.entities import PyramidIdx, YoloBoxes, ImageBatch
from object_detection.models.anchors import Anchors
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize(
    "fsize , size, scales, ratios",
    [
        (4, 2, [0.5 ** (1 / 2), 1.0, 2.0 ** (1 / 2)], [1.0, 1 / 2, 2 / 1]),
        (8, 2, [0.5 ** (1 / 2), 1.0, 2.0 ** (1 / 2)], [1.0, 1 / 2, 2 / 1]),
        (16, 2, [0.5 ** (1 / 2), 1.0, 2.0 ** (1 / 2)], [1.0, 1 / 2, 2 / 1]),
        (32, 2, [0.5 ** (1 / 2), 1.0, 2.0 ** (1 / 2)], [1.0, 1 / 2, 2 / 1]),
        (64, 4, [0.5 ** (1 / 2), 1.0, 2.0 ** (1 / 2)], [1.0, 1 / 2, 2 / 1]),
        (64, 1, [1.0], [1.0, 2 / 3, 3 / 2]),
        (128, 4, [1.0], [1.0, 2 / 3, 3 / 2]),
        (256, 4, [1.0], [1.0, 2 / 3, 3 / 2]),
    ],
)
def test_anchors(
    fsize: int, size: float, scales: List[float], ratios: List[float]
) -> None:
    h = fsize
    w = fsize
    images = ImageBatch(torch.zeros((1, 3, h, w), dtype=torch.float32))
    fn = Anchors(size=size, scales=scales, ratios=ratios)
    res = fn(images)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    offset = num_anchors * h * (w // 2) + num_anchors * h // 2
    ids = [offset + x for x in range(num_anchors)]
    plot = DetectionPlot(w=256, h=256)
    plot.with_yolo_boxes(YoloBoxes(res[ids]), color="red")
    plot.save(
        f"store/test-anchors-{fsize}-{size}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png"
    )
