import torch
import pytest
from typing import Any
from object_detection.entities import YoloBoxes
from object_detection.models.centernetv1 import (
    MkMaps,
    Heatmaps,
    Sizemap,
    DiffMap,
    ToBoxes,
    CenterNetV1,
    Anchors,
    ToBoxes,
    BoxDiffs,
)
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.utils import DetectionPlot


def test_ctdtv1() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    fn = CenterNetV1(
        channels=channels,
        backbone=backbone,
        out_idx=6,
        anchors=Anchors(scales=[1.0], ratios=[1.0]),
    )
    anchors, box_diffs, heatmaps = fn(inputs)
    assert heatmaps.shape == (1, 1, 1024 // 16, 1024 // 16)
    assert anchors.shape == (4096, 4)
    assert anchors.shape == box_diffs.shape[1:]


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkMaps(sigma=2.0)
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors(ratios=[1.0], scales=[1.0])
    anchors = mk_anchors(hm)
    diff_boxes = BoxDiffs(torch.zeros((1, *anchors.shape)))
    diff = in_boxes[0] - anchors[cy * w + cx]
    diff_boxes[0, cy * w + cx] = diff

    out_box_batch, out_conf_batch = to_boxes((anchors, diff_boxes, hm))
    out_boxes = out_box_batch[0]
    assert out_boxes[0, 0] == in_boxes[0, 0]
    assert out_boxes[0, 1] == in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"/store/test-heatmapv1.png")
