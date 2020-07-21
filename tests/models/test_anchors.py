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
    "scales, ratios", [
        ([10.0, 20.0, 40.0], [1.0, 3/4, 4/3]),
        ([10.0, 20.0], [1.0, 1.0/2.0, 2.0/1.0]),
        ([10.0], [1.0, 1.0/2.0, 2.0/1.0]),
    ],
)
def test_anchors(scales: List[float], ratios: List[float]) -> None:
    h = 100
    w = 100
    images = ImageBatch(torch.zeros((1, 3, h, w), dtype=torch.float32))
    fn = Anchors(scales=scales, ratios=ratios)
    res = fn(images)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    offset = num_anchors * h * (w // 2) + num_anchors * h // 2
    ids = [offset + x for x in range(num_anchors)]
    plot = DetectionPlot(w=w, h=h)
    plot.with_yolo_boxes(YoloBoxes(res[ids]), color="red")
    plot.save(f"store/test-anchors-{num_anchors}.png")
