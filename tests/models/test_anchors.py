import torch
import pytest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import List
from object_detection.entities import PyramidIdx
from object_detection.entities.box import YoloBoxes
from object_detection.models.anchors import Anchors
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize(
    "scales, ratios",
    [([1.0], [1.0]), ([1.0], [0.5, 1.0]), ([0.5, 1, 2], [0.5, 1.0, 1.5]),],
)
def test_anchors(scales: List[float], ratios: List[float]) -> None:
    h = 128
    w = 128
    images = torch.zeros((1, 3, h, w), dtype=torch.float32)
    fn = Anchors(size=16, scales=scales, ratios=ratios)
    res = fn(images)
    print(res.shape)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    offset = res.shape[0] // 2 + 1 * num_anchors
    plot = DetectionPlot(w=w, h=h)
    plot.with_yolo_boxes(YoloBoxes(res[offset : offset + num_anchors]), color="red")
    plot.with_yolo_boxes(YoloBoxes(res[offset + num_anchors : offset + num_anchors*2]), color="blue")
    plot.save(f"/store/test-anchors-{num_anchors}.png")
