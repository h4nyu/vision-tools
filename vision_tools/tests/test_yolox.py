from torch import Tensor
import pytest
import torch
from torch import optim
from vision_tools.yolox import (
    YOLOXHead,
    DecoupledHead,
    YOLOX,
    Criterion,
    TrainBatch,
    TrainStep,
)
from vision_tools.backbone import CSPDarknet
from vision_tools.neck import CSPPAFPN
from vision_tools.assign import SimOTA
from vision_tools.utils import ToDevice


@pytest.fixture
def model() -> YOLOX:
    backbone = CSPDarknet()
    num_classes = 2
    box_feat_range = (2, 7)
    patch_size = 128
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
        patch_size=patch_size,
        box_iou_threshold=0.1,
        score_threshold=0.0,
    )


@pytest.fixture
def assign() -> SimOTA:
    return SimOTA(topk=10)


@pytest.fixture
def criterion(assign: SimOTA, model: YOLOX) -> Criterion:
    return Criterion(model=model, assign=assign)


@pytest.fixture
def inputs() -> TrainBatch:
    image_batch = torch.rand(1, 3, 128, 128)
    gt_box_batch = [
        torch.tensor([[10, 10, 20, 20]]),
    ]
    gt_label_batch = [torch.zeros(len(m)).long() for m in gt_box_batch]
    return TrainBatch(
        image_batch=image_batch,
        gt_box_batch=gt_box_batch,
        gt_label_batch=gt_label_batch,
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
) -> None:
    criterion(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_train_step(
    criterion: Criterion,
    model: YOLOX,
    inputs: TrainBatch,
) -> None:
    model.to("cuda")
    inputs = ToDevice("cuda")(**inputs)
    optimizer = optim.Adam(model.parameters())
    train_step = TrainStep(criterion=criterion, optimizer=optimizer, use_amp=False)
    train_step(inputs)
