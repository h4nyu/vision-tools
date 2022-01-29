import pytest
import torch
import os
import PIL
from typing import Dict, List, Tuple
import numpy as np
from toolz.curried import pipe, partition, map, filter
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from cots_bench.yolox import (
    get_model,
    get_criterion,
    get_checkpoint,
    get_writer,
    get_tta_inference_one,
    get_to_boxes,
)
from vision_tools.assign import SimOTA
from torch.utils.data import DataLoader
from vision_tools.yolox import YOLOX, Criterion, ToBoxes
from vision_tools.interface import TrainBatch
from vision_tools.utils import draw, batch_draw, seed_everything, load_config, ToDevice
from vision_tools.batch_transform import BatchRemovePadding
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    collate_fn,
    InferenceTransform,
    read_train_rows,
    kfold,
    Row,
    filter_empty_boxes,
)


cfg = load_config(os.path.join(os.path.dirname(__file__), "../config/yolox.yaml"))
cfg["device"] = "cpu"
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
def to_device() -> ToDevice:
    return ToDevice(cfg["device"])


@pytest.fixture
def to_boxes() -> ToBoxes:
    cfg["to_boxes"]["conf_threshold"] = 0.01
    return get_to_boxes(cfg)


@pytest.fixture
def rows() -> List[Row]:
    return filter_empty_boxes(read_train_rows(cfg["dataset_dir"]))


@pytest.fixture
def batch(rows: List[Row]) -> TrainBatch:
    (
        _,
        rows,
    ) = kfold(rows, cfg["n_splits"], cfg["fold"])
    rows = pipe(rows, filter(lambda x: len(x["boxes"]) > 4), list)
    dataset = COTSDataset(
        rows[10:],
        transform=Transform(cfg),
    )
    loader_iter = iter(DataLoader(dataset, collate_fn=collate_fn, batch_size=1))
    return next(loader_iter)


@torch.no_grad()
def test_assign(
    batch: TrainBatch,
    model: YOLOX,
    criterion: Criterion,
    to_device: ToDevice,
    to_boxes: ToBoxes,
) -> None:
    model.eval()
    batch = to_device(**batch)
    gt_box_batch = batch["box_batch"]
    gt_label_batch = batch["label_batch"]
    image_batch = batch["image_batch"]
    pred_yolo_batch = model(image_batch)
    gt_yolo_batch, pos_idx = criterion.prepeare_box_gt(
        model.num_classes, gt_box_batch, gt_label_batch, pred_yolo_batch
    )
    gt_box_batch = [
        box_convert(b[idx], in_fmt="cxcywh", out_fmt="xyxy")
        for b, idx in zip(gt_yolo_batch[:, :, :4], pos_idx)
    ]
    box_batch = [
        box_convert(b[idx], in_fmt="cxcywh", out_fmt="xyxy")
        for b, idx in zip(pred_yolo_batch[:, :, :4], pos_idx)
    ]
    point_batch = [
        b[idx].view(len(b[idx]), 1, 2)
        for b, idx in zip(pred_yolo_batch[:, :, 7:9], pos_idx)
    ]
    plot = batch_draw(
        image_batch=image_batch,
        box_batch=box_batch,
        point_batch=point_batch,
        gt_box_batch=gt_box_batch,
    )
    writer.add_image("assign", plot, 0)
    box_batch = to_boxes(pred_yolo_batch)["box_batch"]
    plot = batch_draw(
        image_batch=image_batch, box_batch=box_batch, gt_box_batch=gt_box_batch
    )
    writer.add_image("assign", plot, 1)
    writer.flush()


def test_criterion(batch: TrainBatch, model: YOLOX, criterion: Criterion) -> None:
    criterion(model, batch)


def test_tta(rows: List[Row], model: YOLOX, to_device: ToDevice) -> None:
    inference_one = get_tta_inference_one(cfg)
    rows = pipe(rows, filter(lambda x: len(x["boxes"]) > 4), list)
    transform = InferenceTransform(cfg)
    for i, row in enumerate(rows[:1]):
        gt_boxes = row["boxes"]
        img_arr = np.array(PIL.Image.open(row["image_path"]))
        img = transform(image=img_arr)["image"] / 255
        img = to_device(image=img)["image"]
        pred = inference_one(img)
        print(pred["boxes"])
        print(pred["image"].shape)
        plot = draw(image=pred["image"], boxes=pred["boxes"], gt_boxes=gt_boxes)
        writer.add_image("tta_inference_one", plot, i)
    writer.flush()
