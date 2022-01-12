import torch
import pytest
from omegaconf import OmegaConf
from typing import Any
import os

from torch.utils.data import DataLoader

# import torchvision, os, torch
from subaru_bench.data import (
    read_train_rows,
    read_test_rows,
    KuzushijiDataset,
    Transform,
    TrainTransform,
    collate_fn,
    save_submission,
    read_code_map,
    SubRow,
)
from vision_tools.utils import batch_draw, draw
from torch.utils.tensorboard import SummaryWriter

cfg = OmegaConf.load("/app/subaru_bench/config/yolox.yaml")
writer = SummaryWriter("/app/runs/test-subaru_bench")

no_volume = not os.path.exists(cfg.root_dir)
reason = "no data volume"

no_volume = not os.path.exists(cfg.root_dir)
if no_volume:
    pytestmark = pytest.mark.skip("no data volume")


@pytest.fixture
def transform() -> Any:
    return Transform(512)


@pytest.fixture
def train_transform() -> Any:
    return TrainTransform(512)


def test_read_train_rows() -> None:
    rows = read_train_rows(cfg.root_dir)
    assert len(rows) == 3605


def test_read_test_rows() -> None:
    rows = read_test_rows(cfg.root_dir)
    assert len(rows) == 1730


def test_dataset(transform: Any) -> None:
    rows = read_train_rows(cfg.root_dir)
    dataset = KuzushijiDataset(rows, transform=transform)
    sample = dataset[0]
    plot = draw(
        image=sample["image"],
        boxes=sample["boxes"],
    )
    writer.add_image("image", plot, 1)
    writer.flush()


def test_aug(train_transform: Any) -> None:
    rows = read_train_rows(cfg.root_dir)
    dataset = KuzushijiDataset(rows, transform=train_transform)
    loader_iter = iter(DataLoader(dataset, batch_size=8, collate_fn=collate_fn))
    for i in range(2):
        batch = next(loader_iter)
        plot = batch_draw(**batch)
        writer.add_image("aug", plot, i)
    writer.flush()


def test_save_submission() -> None:
    rows: list[SubRow] = [
        {
            "id": "aaaa",
            "points": torch.tensor([[10, 20]]),
            "labels": torch.tensor([0]),
        }
    ]
    code_map = read_code_map(os.path.join(cfg.root_dir, "unicode_translation.csv"))
    save_submission(
        rows,
        code_map,
        os.path.join(cfg.root_dir, f"test_sub.csv"),
    )
