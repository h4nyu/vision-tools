import torch
import pytest
from typing import Any, List
import os

from torch.utils.data import DataLoader

from cots_bench.yolox import (
    get_writer,
)
from cots_bench.data import (
    read_train_rows,
    Transform,
    TrainTransform,
    COTSDataset,
    Row,
    collate_fn,
    kfold,
    to_submission_string,
    filter_empty_boxes,
)
from vision_tools.utils import batch_draw, draw, load_config
from vision_tools.batch_transform import BatchMosaic, BatchRelocate
from torch.utils.tensorboard import SummaryWriter
from toolz.curried import pipe, partition, map, filter

cfg = load_config("/app/cots_bench/config/yolox.2.yaml")
writer = get_writer(cfg)

no_volume = not os.path.exists(cfg["dataset_dir"])
reason = "no data volume"

no_volume = not os.path.exists(cfg["dataset_dir"])
if no_volume:
    pytestmark = pytest.mark.skip("no data volume")


@pytest.fixture
def transform() -> Any:
    return Transform(cfg)


@pytest.fixture
def train_transform() -> Any:
    return TrainTransform(cfg)


@pytest.fixture
def rows() -> List[Row]:
    return filter_empty_boxes(read_train_rows(cfg["dataset_dir"]))


def test_batch(train_transform: Any, rows: List[Row]) -> None:
    dataset = COTSDataset(rows, transform=train_transform)
    loader_iter = iter(DataLoader(dataset, batch_size=4, collate_fn=collate_fn))
    mosaic = BatchMosaic()
    for i in range(1):
        batch = next(loader_iter)
        batch = mosaic(batch)
        plot = batch_draw(
            image_batch=batch["image_batch"], box_batch=batch["box_batch"]
        )
        writer.add_image("batch-aug", plot, i)
    writer.flush()


def test_aug(train_transform: Any, rows: List[Row]) -> None:
    rows = pipe(rows, filter(lambda x: len(x["boxes"]) == 5), list)
    dataset = COTSDataset(
        rows,
        transform=train_transform,
    )
    for i in range(20):
        sample, _ = dataset[0]
        assert sample["image"].shape == (
            3,
            cfg["image_height"],
            cfg["image_width"],
        )
        plot = draw(image=sample["image"], boxes=sample["boxes"])
        writer.add_image("aug", plot, i)
    writer.flush()


def test_to_submission_string() -> None:
    boxes = torch.tensor(
        [
            [0, 0, 100, 100],
            [10, 20, 30, 40],
        ]
    )
    confs = torch.tensor([0.9, 0.5])
    encoded = to_submission_string(boxes, confs)
    expected = [
        0.9,
        0.0,
        0.0,
        100.0,
        100.0,
        0.5,
        10.0,
        20.0,
        20.0,
        20.0,
    ]
    for v, e in zip(encoded.split(" "), expected):
        assert float(v) == pytest.approx(e, abs=1e-3)


def test_fold(rows: List[Row]) -> None:
    train_rows, test_rows = kfold(rows, cfg["n_splits"])
    train_groups = pipe(train_rows, map(lambda x: x["sequence"]), set)
    test_groups = pipe(test_rows, map(lambda x: x["sequence"]), set)
    for i in train_groups:
        assert i not in test_groups
