import pytest
from omegaconf import OmegaConf
from typing import Any
import os

# import torchvision, os, torch
from kuzushiji_bench.data import (
    read_train_rows,
    read_test_rows,
    KuzushijiDataset,
    Transfrom,
    # inv_normalize,
    # train_transforms,
    # kfold,
    # save_submission,
    # read_code_map,
    # SubRow,
)
from vision_tools.utils import draw_save

config = OmegaConf.load("/app/kuzushiji_bench/config/dataset.yaml")

no_volume = not os.path.exists(config.root_dir)
reason = "no data volume"


@pytest.fixture
def transforms() -> Any:
    return Transfrom(512)


@pytest.mark.skipif(no_volume, reason=reason)
def test_read_train_rows() -> None:
    rows = read_train_rows(config.root_dir)
    assert len(rows) == 3605


@pytest.mark.skipif(no_volume, reason=reason)
def test_read_test_rows() -> None:
    rows = read_test_rows(config.root_dir)
    assert len(rows) == 1730


@pytest.mark.skipif(no_volume, reason=reason)
def test_dataset(transforms: Any) -> None:
    rows = read_train_rows(config.root_dir)
    dataset = KuzushijiDataset(rows, transforms=transforms)
    sample = dataset[0]
    draw_save(
        "/app/test_outputs/test-kuzushiji.png",
        image=sample["image"],
        boxes=sample["boxes"],
    )


# def test_aug() -> None:
#     rows = read_train_rows(config.root_dir)
#     dataset = KuzushijiDataset(rows, transforms=train_transforms)
#     for i in range(3):
#         sample = dataset[100]
#         img, boxes, labels, _, _ = sample


# def test_fold() -> None:
#     rows = read_train_rows(config.root_dir)
#     a, b = kfold(rows, n_splits=4)


# def test_save_submission() -> None:
#     rows: list[SubRow] = [
#         {
#             "id": "aaaa",
#             "points": torch.tensor([[10, 20]]),
#             "labels": torch.tensor([0]),
#         }
#     ]
#     code_map = read_code_map(os.path.join(config.root_dir, "unicode_translation.csv"))
#     save_submission(
#         rows,
#         code_map,
#         os.path.join(config.root_dir, f"test_sub.csv"),
#     )
