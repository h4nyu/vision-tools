import torch
import pytest
from omegaconf import OmegaConf
from typing import Any
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
)
from vision_tools.utils import batch_draw, draw
from torch.utils.tensorboard import SummaryWriter

cfg = OmegaConf.load("/app/cots_bench/config/yolox.yaml")
writer = SummaryWriter("/app/runs/test-cots_bench")

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


@pytest.fixture
def train_rows() -> list[Row]:
    return read_train_rows(cfg.root_dir)


def test_aug(train_transform: Any, train_rows: list[Row]) -> None:
    dataset = COTSDataset(
        train_rows, transform=train_transform, image_dir=cfg.image_dir
    )
    loader_iter = iter(DataLoader(dataset, batch_size=8, collate_fn=collate_fn))
    for i in range(2):
        batch = next(loader_iter)
        plot = batch_draw(**batch)
        writer.add_image("aug", plot, i)
    writer.flush()


# def test_save_submission() -> None:
#     rows: list[SubRow] = [
#         {
#             "id": "aaaa",
#             "points": torch.tensor([[10, 20]]),
#             "labels": torch.tensor([0]),
#         }
#     ]
#     code_map = read_code_map(os.path.join(cfg.root_dir, "unicode_translation.csv"))
#     save_submission(
#         rows,
#         code_map,
#         os.path.join(cfg.root_dir, f"test_sub.csv"),
#     )
