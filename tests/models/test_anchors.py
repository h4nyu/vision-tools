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
    "fsize ,scales, ratios", [
        (64, [6.0, 9.0], [1.0, 2/3, 3/2]),
        (128, [6.0, 9.0], [1.0, 2/3, 3/2]),
        (256, [6.0, 9.0], [1.0, 2/3, 3/2]),
    ],
)
def test_anchors(fsize:int, scales: List[float], ratios: List[float]) -> None:
    h = fsize
    w = fsize
    images = ImageBatch(torch.zeros((1, 3, h, w), dtype=torch.float32))
    fn = Anchors(scales=scales, ratios=ratios)
    res = fn(images)
    num_anchors = len(scales) * len(ratios)
    anchor_count = w * h * num_anchors
    assert res.shape == (anchor_count, 4)
    offset = num_anchors * h * (w // 2) + num_anchors * h // 2
    ids = [offset + x for x in range(num_anchors)]
    plot = DetectionPlot(w=512, h=512)
    plot.with_yolo_boxes(YoloBoxes(res[ids]), color="red")
    plot.save(f"store/test-anchors-{fsize}-{'-'.join([str(x) for x in  scales])}-{num_anchors}.png")
