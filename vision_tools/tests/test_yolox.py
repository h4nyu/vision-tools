from typing import Any, Dict, Iterator, Tuple

import pytest
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vision_tools.assign import SimOTA
from vision_tools.backbone import CSPDarknet
from vision_tools.meter import MeanReduceDict
from vision_tools.neck import CSPPAFPN
from vision_tools.utils import ToDevice
from vision_tools.yolox import YOLOX, Criterion, DecoupledHead, TrainBatch, YOLOXHead


@pytest.fixture
def model() -> YOLOX:
    backbone = CSPDarknet()
    num_classes = 2
    feat_range = (2, 6)
    head_range = (1, 3)
    neck = CSPPAFPN(
        in_channels=backbone.channels[feat_range[0] : feat_range[1]],
        strides=backbone.strides[feat_range[0] : feat_range[1]],
    )
    return YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=64,
        num_classes=num_classes,
        feat_range=feat_range,
        head_range=head_range,
    )


@pytest.fixture
def assign() -> SimOTA:
    return SimOTA(topk=10)


@pytest.fixture
def criterion(assign: SimOTA) -> Criterion:
    return Criterion(assign=assign)


@pytest.fixture
def loader(inputs: TrainBatch) -> Any:
    return [inputs]


@pytest.fixture
def writer() -> SummaryWriter:
    return SummaryWriter()


@pytest.fixture
def inputs() -> TrainBatch:
    image_batch = torch.rand(1, 3, 128, 128 * 2)
    box_batch = [
        torch.tensor([[10, 10, 20, 20]]),
    ]
    label_batch = [torch.zeros(len(m)).long() for m in box_batch]
    conf_batch = [torch.ones(len(m)).float() for m in box_batch]
    return TrainBatch(
        image_batch=image_batch,
        box_batch=box_batch,
        label_batch=label_batch,
        conf_batch=conf_batch,
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
    image_size = 128
    images = torch.rand(2, 3, image_size * 2, image_size)
    feats = model.feats(images)
    yolo_batch = model.box_branch(feats)
    assert yolo_batch.shape[0] == 2
    assert yolo_batch.shape[2] == 5 + model.num_classes + 3


def test_criterion(
    criterion: Criterion,
    inputs: TrainBatch,
    model: YOLOX,
) -> None:
    criterion(model, inputs)
