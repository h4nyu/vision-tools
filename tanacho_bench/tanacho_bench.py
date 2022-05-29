from __future__ import annotations

import json
import os
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import PIL
import torch
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing
from torch import Tensor, nn, optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image


def preprocess(path: str, image_dir: str) -> dict:
    rows = []
    with open(path, "r") as f:
        meta = json.load(f)
    for key in meta:
        filenames = os.listdir(os.path.join(image_dir, key))
        for filename in filenames:
            row = {}
            row["part_id"] = key
            row["category"] = meta[key]["category"]
            row["color"] = meta[key]["color"]
            row["image_path"] = os.path.join(image_dir, key, filename)
            rows.append(row)
    category_le = preprocessing.LabelEncoder()
    category_le.fit([row["category"] for row in rows])
    category_labels = category_le.transform([row["category"] for row in rows])
    color_le = preprocessing.LabelEncoder()
    color_le.fit([row["color"] for row in rows])
    color_labels = color_le.transform([row["color"] for row in rows])
    for i, row in enumerate(rows):
        row["category_label"] = category_labels[i]
        row["color_label"] = color_labels[i]
    encoders = dict(
        category=category_le,
        color=color_le,
    )
    res = dict(
        rows=rows,
        encoders=encoders,
    )
    return res


def eda(rows: list[dict]) -> None:
    print("Number of rows:", len(rows))
    print("Number of unique categories:", len(set([row["category"] for row in rows])))
    print("Number of unique colors:", len(set([row["color"] for row in rows])))
    print("Number of unique part ids:", len(set([row["part_id"] for row in rows])))


TrainTransform = lambda cfg: A.Compose(
    [
        ToTensorV2(),
    ],
)


class TanachoDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        transform: Callable,
    ) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], dict]:
        row = self.rows[idx]
        im = PIL.Image.open(row["image_path"])
        img_arr = np.array(im)
        transformed = self.transform(
            image=img_arr,
        )
        image = (transformed["image"] / 255).float()
        category_label = torch.tensor(row["category_label"] or 0)
        color_label = torch.tensor(row["color_label"] or 0)
        sample = dict(
            image=image,
            category_label=category_label,
            color_label=color_label,
        )
        return sample, row


def preview_dataset(cfg: dict, rows: list[dict], path: str) -> None:
    dataset = TanachoDataset(
        rows=rows,
        transform=TrainTransform(cfg),
    )
    grid = make_grid([dataset[i][0]["image"] for i in range(4)])
    save_image(grid, path)
