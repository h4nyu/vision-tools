from typing import Iterator
import pytest
import torch
from torch import optim
from vision_tools.yolox import (
    YOLOXHead,
    DecoupledHead,
    YOLOX,
    Criterion,
    TrainBatch,
)
from vision_tools.backbone import CSPDarknet
from vision_tools.neck import CSPPAFPN
from vision_tools.assign import SimOTA
from vision_tools.utils import ToDevice
from vision_tools.step import TrainStep


@pytest.fixture
def model() -> YOLOX:
    backbone = CSPDarknet()
    num_classes = 2
    box_feat_range = (3, 6)
    neck = CSPPAFPN(
        in_channels=backbone.channels,
        strides=backbone.strides,
    )
    return YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=64,
        num_classes=num_classes,
        box_feat_range=box_feat_range,
        box_iou_threshold=0.1,
        score_threshold=0.0,
    )


@pytest.fixture
def assign() -> SimOTA:
    return SimOTA(topk=10)


@pytest.fixture
def criterion(assign: SimOTA) -> Criterion:
    return Criterion(assign=assign)


@pytest.fixture
def inputs() -> TrainBatch:
    image_batch = torch.rand(1, 3, 128, 128)
    box_batch = [
        torch.tensor([[10, 10, 20, 20]]),
    ]
    label_batch = [torch.zeros(len(m)).long() for m in box_batch]
    return TrainBatch(
        image_batch=image_batch,
        box_batch=box_batch,
        label_batch=label_batch,
    )


def test_yoloxhead() -> None:
    in_channels = [
        32,
        64,
        128,
    ]
    hidden_channels = 48
    num_classes = 3
    head = YOLOXHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
    )
    feats = [torch.rand(2, c, 32, 32) for c in in_channels]
    res = head(feats)
    assert len(feats) == len(res)
    for f, r in zip(feats, res):
        assert r.size() == (2, 4 + 1 + num_classes, r.size(2), r.size(3))


@pytest.mark.parametrize("depthwise", [True, False])
def test_decoupled_head(depthwise: bool) -> None:
    in_channels = 32
    hidden_channels = 48
    num_classes = 3
    head = DecoupledHead(
        num_classes=num_classes,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        depthwise=depthwise,
    )
    feat = torch.rand(2, in_channels, 32, 32)
    res = head(feat)
    assert res.size() == (2, 4 + 1 + num_classes, 32, 32)


def test_box_branch(model: YOLOX) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    feats = model.feats(images)
    box_feats = model.box_feats(feats)
    yolo_batch = model.box_branch(box_feats)
    assert yolo_batch.shape[0] == 2
    assert yolo_batch.shape[2] == 5 + model.num_classes + 3


def test_criterion(
    criterion: Criterion,
    inputs: TrainBatch,
    model: YOLOX,
) -> None:
    criterion(model, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_train_step(
    criterion: Criterion,
    model: YOLOX,
    inputs: TrainBatch,
) -> None:
    model.to("cuda")
    optimizer = optim.Adam(model.parameters())

    def loader() -> Iterator[TrainBatch]:
        yield inputs

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=ToDevice("cuda"),
        criterion=criterion,
        optimizer=optimizer,
        loader=loader(),
    )
    train_step(model)
