from typing import *
import torchvision, os, torch
from bench.kuzushiji.data import (
    read_train_rows,
    read_test_rows,
    KuzushijiDataset,
    inv_normalize,
    train_transforms,
    kfold,
    save_submission,
    read_code_map,
    SubRow,
)
from vision_tools.utils import DetectionPlot
from vision_tools import Labels, Points
from bench.kuzushiji import config


def test_read_train_rows() -> None:
    rows = read_train_rows(config.root_dir)
    assert len(rows) == 3605


def test_read_test_rows() -> None:
    rows = read_test_rows(config.root_dir)
    assert len(rows) == 1730


def test_dataset() -> None:
    rows = read_train_rows(config.root_dir)
    dataset = KuzushijiDataset(rows)
    sample = dataset[0]
    img, boxes, labels, _, _ = sample
    plot = DetectionPlot(inv_normalize(img))
    plot.draw_boxes(boxes)
    plot.save(os.path.join(config.root_dir, "test_dataset.png"))


def test_aug() -> None:
    rows = read_train_rows(config.root_dir)
    dataset = KuzushijiDataset(rows, transforms=train_transforms)
    for i in range(3):
        sample = dataset[100]
        img, boxes, labels, _, _ = sample
        plot = DetectionPlot(inv_normalize(img))
        plot.draw_boxes(boxes)
        plot.save(os.path.join(config.root_dir, f"test_aug{i}.png"))


def test_fold() -> None:
    rows = read_train_rows(config.root_dir)
    a, b = kfold(rows, n_splits=4)


def test_save_submission() -> None:
    rows: List[SubRow] = [
        {
            "id": "aaaa",
            "points": Points(torch.tensor([[10, 20]])),
            "labels": Labels(torch.tensor([0])),
        }
    ]
    code_map = read_code_map(os.path.join(config.root_dir, "unicode_translation.csv"))
    save_submission(
        rows,
        code_map,
        os.path.join(config.root_dir, f"test_sub.csv"),
    )
