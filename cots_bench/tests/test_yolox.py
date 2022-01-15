import pytest
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from cots_bench.yolox import get_model, get_criterion, get_checkpoint, get_writer
from vision_tools.assign import SimOTA
from torch.utils.data import DataLoader
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.interface import TrainBatch
from vision_tools.utils import draw, batch_draw, seed_everything, load_config
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    collate_fn,
    read_train_rows,
    kfold,
)


cfg = load_config(os.path.join(os.path.dirname(__file__), "../config/yolox.yaml"))
writer = get_writer(cfg)

seed_everything()

no_volume = not os.path.exists(cfg["dataset_dir"])
if no_volume:
    pytestmark = pytest.mark.skip("no data volume")


@pytest.fixture
def model() -> YOLOX:
    m = get_model(cfg)
    checkpoint = get_checkpoint(cfg)
    checkpoint.load_if_exists(
        m,
    )
    return m


@pytest.fixture
def criterion() -> Criterion:
    return get_criterion(cfg)


@pytest.fixture
def batch() -> TrainBatch:
    _, rows = kfold(read_train_rows(cfg["dataset_dir"]), cfg["n_splits"])
    dataset = COTSDataset(
        rows[10:],
        transform=Transform(cfg["image_size"]),
        dataset_dir=cfg["dataset_dir"],
    )
    loader_iter = iter(DataLoader(dataset, collate_fn=collate_fn, batch_size=1))
    return next(loader_iter)


@torch.no_grad()
def test_assign(batch: TrainBatch, model: YOLOX, criterion: Criterion) -> None:
    model.eval()
    gt_box_batch = batch["box_batch"]
    gt_label_batch = batch["label_batch"]
    image_batch = batch["image_batch"]
    feats = model.feats(image_batch)
    pred_yolo_batch = model.box_branch(feats)
    gt_yolo_batch, pos_idx = criterion.prepeare_box_gt(
        model.num_classes, gt_box_batch, gt_label_batch, pred_yolo_batch
    )
    box_batch = [
        box_convert(b[idx], in_fmt="cxcywh", out_fmt="xyxy")
        for b, idx in zip(gt_yolo_batch[:, :, :4], pos_idx)
    ]
    plot = batch_draw(image_batch=image_batch, box_batch=box_batch)
    writer.add_image("assign", plot, 0)
    box_batch = [
        box_convert(b[idx], in_fmt="cxcywh", out_fmt="xyxy")
        for b, idx in zip(pred_yolo_batch[:, :, :4], pos_idx)
    ]
    point_batch = [
        b[idx].view(len(b[idx]), 1, 2)
        for b, idx in zip(pred_yolo_batch[:, :, 7:9], pos_idx)
    ]
    plot = batch_draw(
        image_batch=image_batch, box_batch=box_batch, point_batch=point_batch
    )
    writer.add_image("assign", plot, 1)
    score_batch, box_batch, label_batch = model.to_boxes(pred_yolo_batch)
    plot = batch_draw(image_batch=image_batch, box_batch=box_batch)
    writer.add_image("assign", plot, 2)
    writer.flush()


def test_criterion(batch: TrainBatch, model: YOLOX, criterion: Criterion) -> None:
    criterion(model, batch)
