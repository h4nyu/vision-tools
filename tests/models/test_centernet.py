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
from object_detection.entities import YoloBoxes, Image, ImageSize
from object_detection.entities.box import yolo_to_coco
from object_detection.utils import DetectionPlot
from object_detection.data.object import ObjectDataset
from torch.utils.data import DataLoader


def test_trainer(mocker: Any) -> None:
    train_dataset = ObjectDataset(
        (256, 128), object_size_range=(32, 64), num_samples=128
    )
    test_dataset = ObjectDataset((256, 128), object_size_range=(32, 64), num_samples=8)
    model = CenterNet()
    model_loader = mocker.Mock()
    model_loader.model = model
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(
        DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8),
        DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8),
        model_loader,
        optimizer,
        Visualize("/store", "test"),
        "cuda",
    )
    trainer.train(2)


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
    inputs = torch.rand((1, 3, 1024, 1024))
    fn = CenterNet()
    heatmap, sizemap = fn(inputs)
    assert heatmap.shape == (1, 1, 1024 // 2, 1024 // 2)


@pytest.mark.parametrize("h, w, cy, cx, dy, dx", [(40, 40, 16, 8, 0.001, 0.002)])
def test_mkmaps(h: int, w: int, cy: int, cx: int, dy: float, dx: float) -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.201, 0.402, 0.1, 0.3]]))
    to_boxes = ToBoxes(thresold=0.1)
    mkmaps = MkMaps(sigma=0.3)
    hm, sm, dm = mkmaps([in_boxes], (h, h), (h * 10, w * 10))
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert (sm.nonzero()[0, 2:] - torch.tensor([[cy, cx]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, h, w)
    assert sm.shape == (1, 2, h, w)
    assert (sm[0, :, cy, cx] - torch.tensor([0.1, 0.3])).sum() == 0
    assert (dm[0, :, cy, cx] - torch.tensor([dx, dy])).sum().abs() < 1e-7
    out_boxes, _ = next(iter(to_boxes((hm, sm, dm))))
    assert out_boxes[0, 0] == in_boxes[0, 0]
    assert out_boxes[0, 1] == in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot(w=w, h=h)
    plot.with_image((hm[0, 0] + 1e-4).log())
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes)
    plot.save(f"/store/test-soft-heatmap.png")
