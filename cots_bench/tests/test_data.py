import torch
import pytest
from typing import Any, List
import os

from torch.utils.data import DataLoader

# import torchvision, os, torch
from cots_bench.data import (
    read_train_rows,
    Transform,
    TrainTransform,
    COTSDataset,
    Row,
    collate_fn,
    kfold,
    to_submission_string,
)
from vision_tools.utils import batch_draw, draw, load_config
from torch.utils.tensorboard import SummaryWriter
from toolz.curried import pipe, partition, map, filter

cfg = load_config("/app/cots_bench/config/yolox.yaml")
writer = SummaryWriter("/app/runs/test-cots_bench")

no_volume = not os.path.exists(cfg["dataset_dir"])
reason = "no data volume"

no_volume = not os.path.exists(cfg["dataset_dir"])
if no_volume:
    pytestmark = pytest.mark.skip("no data volume")


@pytest.fixture
def transform() -> Any:
    return Transform(512)


@pytest.fixture
def train_transform() -> Any:
    return TrainTransform(512)


@pytest.fixture
def rows() -> List[Row]:
    return read_train_rows(cfg["dataset_dir"])


def test_aug(train_transform: Any, rows: List[Row]) -> None:
    dataset = COTSDataset(rows, transform=train_transform)
    loader_iter = iter(DataLoader(dataset, batch_size=8, collate_fn=collate_fn))
    for i in range(1):
        batch = next(loader_iter)
        plot = batch_draw(**batch)
        writer.add_image("aug", plot, i)
    writer.flush()


def test_to_submission_string() -> None:
    boxes = torch.tensor([
        [0, 0, 100, 100],
        [10, 20, 30, 40],
    ])
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
