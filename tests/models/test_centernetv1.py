import torch
import pytest
import torch.nn.functional as F
from typing import Any
from object_detection.entities import YoloBoxes, BoxMaps, boxmap_to_boxes, ImageBatch
from object_detection.models.centernetv1 import (
    Heatmaps,
    ToBoxes,
    CenterNetV1,
    Anchors,
    ToBoxes,
    HFlipTTA,
    MkCrossMaps,
    MkGaussianMaps,
    MkFillMaps,
    NearnestAssign,
    MkCornerMaps,
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
    fn = CenterNetV1(
        channels=channels,
        backbone=backbone,
        out_idx=6,
    )
    anchors, box_diffs, heatmaps = fn(inputs)
    assert heatmaps.shape == (1, 1, 512 // 16, 512 // 16)
    assert anchors.shape == (4, 512 // 16, 512 // 16)
    assert anchors.shape == box_diffs.shape[1:]


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkGaussianMaps(sigma=2.0)
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert (torch.nonzero(hm.eq(1), as_tuple=False)[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors()
    anchormap = mk_anchors(hm)
    diffmaps = BoxMaps(torch.zeros((1, *anchormap.shape)))
    diffmaps = in_boxes.view(1, 4, 1, 1).expand_as(diffmaps) - anchormap

    out_box_batch, out_conf_batch = to_boxes((anchormap, diffmaps, hm))
    out_boxes = out_box_batch[0]
    for box in out_boxes:
        assert F.l1_loss(box, in_boxes[0]) < 1e-8
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-heatmapv1.png")


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkfillmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkFillMaps(sigma=0.5)
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors()
    anchormap = mk_anchors(hm)
    diffmaps = BoxMaps(torch.zeros((1, *anchormap.shape)))
    diffmaps = in_boxes.view(1, 4, 1, 1).expand_as(diffmaps) - anchormap

    out_box_batch, out_conf_batch = to_boxes((anchormap, diffmaps, hm))
    out_boxes = out_box_batch[0]
    for box in out_boxes:
        assert F.l1_loss(box, in_boxes[0]) < 1e-8
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.save(f"store/test-heatmapv1-fill.png")


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(80, 80, 32, 16, 0.001, 0.002)])
def test_mkcornermaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkCornerMaps()
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors()
    anchormap = mk_anchors(hm)
    diffmaps = BoxMaps(torch.zeros((1, *anchormap.shape)))
    diffmaps = in_boxes.view(1, 4, 1, 1).expand_as(diffmaps) - anchormap

    out_box_batch, out_conf_batch = to_boxes((anchormap, diffmaps, hm))
    out_boxes = out_box_batch[0]
    for box in out_boxes:
        assert F.l1_loss(box, in_boxes[0]) < 1e-8
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.save(f"store/test-corner.png")


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(80, 80, 32, 16, 0.001, 0.002)])
def test_mkcrossmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkCrossMaps()
    hm = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 1, h, w)
    mk_anchors = Anchors()
    anchormap = mk_anchors(hm)
    diffmaps = BoxMaps(torch.zeros((1, *anchormap.shape)))
    diffmaps = in_boxes.view(1, 4, 1, 1).expand_as(diffmaps) - anchormap

    out_box_batch, out_conf_batch = to_boxes((anchormap, diffmaps, hm))
    out_boxes = out_box_batch[0]
    for box in out_boxes:
        assert F.l1_loss(box, in_boxes[0]) < 1e-8
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.save(f"store/test-crossmap.png")


def test_hfliptta() -> None:
    to_boxes = ToBoxes(threshold=0.1)
    fn = HFlipTTA(to_boxes)
    images = ImageBatch(torch.zeros((1, 3, 64, 64)))
    images[:, :, 0, 0] = torch.ones((1, 3))

    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    model = CenterNetV1(
        channels=channels,
        backbone=backbone,
        out_idx=6,
    )
    fn(model, images)


def test_nearest_assign() -> None:
    x = YoloBoxes(
        torch.tensor(
            [
                [0.11, 0.12, 0.1, 0.1],
                [0.21, 0.22, 0.1, 0.1],
                [0.25, 0.22, 0.1, 0.1],
                [0.31, 0.32, 0.1, 0.1],
                [0.41, 0.42, 0.1, 0.1],
                [0.61, 0.62, 0.1, 0.1],
            ]
        )
    )

    y = YoloBoxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.1, 0.1],
                [0.3, 0.3, 0.1, 0.1],
            ]
        )
    )

    fn = NearnestAssign()
    matched_idx, positive_idx = fn(x, y)
    assert (
        F.l1_loss(matched_idx.float(), torch.tensor([0, 1, 1, 2, 2, 2]).float()) < 1e-9
    )
    assert (
        F.l1_loss(positive_idx.float(), torch.tensor([1, 1, 0, 1, 0, 0]).float()) < 1e-9
    )
