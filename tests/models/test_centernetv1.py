import torch
import pytest
from typing import Any
from object_detection.entities import YoloBoxes
from object_detection.models.centernetv1 import MkMaps, Heatmap, Sizemap, DiffMap, ToBoxes
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize(
    "mode, boxes",
    [
        #  (
        #      "aspect",
        #      [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        #  ),
        #  (
        #      "length",
        #      [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        #  ),
        #  (
        #      "length",
        #      [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        #  ),
        ("length", [[0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.3, 0.3],]),
        #  ("length", []),
    ],
)
def test_mkmap_count(mode: Any, boxes: Any) -> None:
    h = 128
    w = h
    in_boxes = YoloBoxes(torch.tensor(boxes))
    to_boxes = ToBoxes()
    mkmaps = MkMaps(sigma=5.0, mode=mode)
    hm, sm, dm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    out_boxes, _ = next(iter(to_boxes((hm, sm, dm))))
    assert hm.eq(1).nonzero().shape == (len(in_boxes), 4)  # type:ignore

    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"/store/test-heatmapv1-{mode}-{len(in_boxes)}.png")

    plot = DetectionPlot(w=w, h=h)
    plot.with_image((dm[0, 0]).abs())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"/store/test-sizemapv1-{mode}-{len(in_boxes)}.png")
