import typing as t
import pytest
import numpy as np
import torch
from typing import Any
from object_detection.models.centernet import (
    CenterNet,
    MkMaps,
    ToBoxes,
    HMLoss,
    Trainer,
    Visualize,
    collate_fn,
)
from object_detection.entities import YoloBoxes, Image, ImageSize, ImageBatch
from object_detection.entities.box import yolo_to_coco
from object_detection.utils import DetectionPlot
from object_detection.models.backbones.resnet import ResNetBackbone
from torch.utils.data import DataLoader


def test_hm_loss() -> None:
    heatmaps = torch.tensor([[0.1, 0.5, 0.1], [0.5, 1, 0.5], [0.1, 0.5, 0.1],])
    fn = HMLoss(beta=4)
    preds = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    res = fn(preds, heatmaps)
    assert res < 1e-4

    preds = torch.tensor([[0.0, 0.0, 0.0], [0.5, 1, 0.5], [0.0, 0.0, 0.0],])
    res = fn(preds, heatmaps)
    assert res < 0.03

    preds = torch.tensor([[0.5, 0.0, 0.5], [0.0, 1, 0.0], [0.0, 0.0, 0.0],])
    res = fn(preds, heatmaps)
    assert res < 0.3


def test_centernet_foward() -> None:
    inputs = torch.rand((1, 3, 512, 512))
    channels = 32
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    fn = CenterNet(channels=channels, backbone=backbone, out_idx=3)
    heatmap, sizemap, _, _ = fn(inputs)
    assert heatmap.shape == (1, 1, 512 // 2, 512 // 2)


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkMaps(sigma=0.3)
    hm, sm, dm, counts = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    assert (torch.nonzero(hm.eq(1), as_tuple=False)[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert (torch.nonzero(sm, as_tuple=False)[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, h, w)
    assert sm.shape == (1, 2, h, w)
    assert (sm[0, :, cy, cx] - torch.tensor([0.1, 0.3])).sum() == 0
    assert (dm[0, :, cy, cx] - torch.tensor([dx, dy])).sum().abs() < 1e-6
    out_boxes, _ = next(iter(to_boxes((hm, sm, dm, counts))))
    assert out_boxes[0, 0] == in_boxes[0, 0]
    assert out_boxes[0, 1] == in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-soft-heatmap.png")


@pytest.mark.parametrize(
    "mode, boxes",
    [
        (
            "aspect",
            [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        ),
        (
            "length",
            [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        ),
        (
            "length",
            [[0.6, 0.6, 0.05, 0.3], [0.4, 0.4, 0.3, 0.3], [0.1, 0.2, 0.1, 0.2],],
        ),
        ("length", [[0.3, 0.3, 0.1, 0.1], [0.4, 0.4, 0.3, 0.3],]),
        ("length", []),
    ],
)
def test_mkmap_count(mode: Any, boxes: Any) -> None:
    h = 128
    w = h
    in_boxes = YoloBoxes(torch.tensor(boxes))
    to_boxes = ToBoxes()
    mkmaps = MkMaps(sigma=5.0, mode=mode)
    hm, sm, dm, counts = mkmaps([in_boxes], (h, w), (h * 10, w * 10))
    out_boxes, _ = next(iter(to_boxes((hm, sm, dm, counts))))
    assert torch.nonzero(hm.eq(1), as_tuple=False).shape == (
        len(in_boxes),
        4,
    )  # type:ignore
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-heatmap-count-{mode}-{len(in_boxes)}.png")
