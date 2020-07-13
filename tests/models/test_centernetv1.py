import torch
import pytest
from typing import Any
from object_detection.entities import YoloBoxes, BoxMaps, boxmap_to_boxes, ImageBatch
from object_detection.models.centernetv1 import (
    MkMaps,
    Heatmaps,
    Sizemap,
    ToBoxes,
    CenterNetV1,
    Anchors,
    ToBoxes,
    HFlipTTA,
)
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.utils import DetectionPlot
from torch import nn


def test_anchors() -> None:
    fn = Anchors(size=2)
    hm = torch.zeros((1, 1, 8, 8))
    anchors = fn(hm)
    boxes = boxmap_to_boxes(anchors)
    assert len(boxes) == 8 * 8

    plot = DetectionPlot(w=8, h=8)
    plot.with_yolo_boxes(YoloBoxes(boxes[[0, 4, 28, 27]]), color="blue")
    plot.save(f"store/test-anchorv1.png")


def test_ctdtv1() -> None:
    inputs = torch.rand((1, 3, 512, 512))
    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    fn = CenterNetV1(channels=channels, backbone=backbone, out_idx=6,)
    anchors, box_diffs, heatmaps = fn(inputs)
    assert heatmaps.shape == (1, 1, 512 // 16, 512 // 16)
    assert anchors.shape == (4, 512 // 16, 512 // 16)
    assert anchors.shape == box_diffs.shape[1:]


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkMaps(sigma=2.0)
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors()
    anchormap = mk_anchors(hm)
    diffmaps = BoxMaps(torch.zeros((1, *anchormap.shape)))
    diff = in_boxes[0] - anchormap[:, cy, cx]
    diffmaps[0, :, cy, cx] = diff

    out_box_batch, out_conf_batch = to_boxes((anchormap, diffmaps, hm))
    out_boxes = out_box_batch[0]
    assert out_boxes[0, 0] == in_boxes[0, 0]
    assert out_boxes[0, 1] == in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-heatmapv1.png")


def test_hfliptta() -> None:
    to_boxes = ToBoxes(threshold=0.1)
    fn = HFlipTTA(to_boxes)
    images = ImageBatch(torch.zeros((1, 3, 64, 64)))
    images[:, :, 0, 0] = torch.ones((1, 3))

    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    model = CenterNetV1(channels=channels, backbone=backbone, out_idx=6,)
    fn(model, images)
