import pytest
import torch
import os
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from kuzushiji_bench.yolox import get_model, get_criterion
from vision_tools.assign import SimOTA
from torch.utils.data import DataLoader
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.interface import TrainBatch
from vision_tools.utils import draw
from kuzushiji_bench.data import (
    KuzushijiDataset,
    TrainTransform,
    collate_fn,
    read_train_rows,
)

cfg = OmegaConf.load("/app/kuzushiji_bench/config/yolox.yaml")
writer = SummaryWriter("/app/runs/test-kuzushiji_bench")

no_volume = not os.path.exists(cfg.root_dir)
reason = "no data volume"


@pytest.fixture
def model() -> YOLOX:
    return get_model(cfg)


@pytest.fixture
def criterion() -> Criterion:
    return get_criterion(cfg)


@pytest.fixture
def batch() -> TrainBatch:
    row = read_train_rows(cfg.root_dir)
    dataset = KuzushijiDataset(
        row[:200],
        transform=TrainTransform(cfg.image_size),
    )
    loader_iter = iter(DataLoader(dataset, collate_fn=collate_fn, batch_size=1))
    return next(loader_iter)


@pytest.mark.skipif(no_volume, reason=reason)
def test_assign(batch: TrainBatch, model: YOLOX, criterion: Criterion) -> None:
    gt_box_batch = batch["box_batch"]
    gt_label_batch = batch["label_batch"]
    image_batch = batch["image_batch"]
    feats = model.feats(image_batch)
    pred_yolo_batch = model.box_branch(feats)
    gt_yolo_batch, pos_idx = criterion.prepeare_box_gt(
        model.num_classes, gt_box_batch, gt_label_batch, pred_yolo_batch
    )
    boxes = box_convert(
        pred_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
    )
    plot = draw(image=image_batch[0], boxes=gt_box_batch[0])
    writer.add_image("assign", plot, 0)
    plot = draw(image=image_batch[0], boxes=boxes)
    writer.add_image("assign", plot, 1)
    writer.flush()

    # draw_save(
    #     "/app/test_outputs/yolox-assign-anchor.png",
    #     images[0],
    #     boxes=box_convert(
    #         pred_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
    #     ),
    # )
    # draw_save(
    #     "/app/test_outputs/yolox-assign-gt.png",
    #     images[0],
    #     boxes=box_convert(
    #         gt_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
    #     ),
    # )
