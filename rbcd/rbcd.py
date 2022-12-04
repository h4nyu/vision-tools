from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import optuna
import pandas as pd
import torch
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset


def pfbeta(labels: Any, predictions: Any, beta: float = 1.0) -> float:
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


@dataclass
class Config:
    name: str
    image_size: int
    seed: int = 42
    n_splits: int = 5

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
            print(obj)
        return Config(**obj)


class RdcdPngDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = f"{self.image_dir}/{row['patient_id']}_{row['image_id']}.png"
        image = io.imread(image_path)
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        target = torch.tensor(row["cancer"]) if row["cancer"] is not None else None
        sample = dict(
            **row,
            image=image,
            target=target,
        )
        return sample


TrainTransform = lambda cfg: A.Compose(
    [
        A.LongestMaxSize(max_size=cfg.image_size),
        A.RandomRotate90(p=cfg.random_rotate_p),
        A.HorizontalFlip(p=cfg.hflip_p),
        A.VerticalFlip(p=cfg.vflip_p),
        ToTensorV2(),
    ],
)


class RdcdDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = f"{self.image_dir}/{row['patient_id']}_{row['image_id']}.png"
        image = io.imread(image_path)
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        target = torch.tensor(row["cancer"]) if row["cancer"] is not None else None
        sample = dict(
            **row,
            image=image,
            target=target,
        )
        return sample


class SetupFolds:
    def __init__(
        self,
        seed: int,
        n_splits: int,
    ) -> None:
        self.seed = seed
        self.n_splits = n_splits
        # self.skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        self.kf = StratifiedGroupKFold(
            n_splits=n_splits, random_state=seed, shuffle=True
        )
        self.folds = []

    # patiant_id, cancer
    def __call__(self, df: pd.Dataframe) -> list[pd.Dataframe]:
        y = df["cancer"].values
        X = df["image_id"].values
        groups = df["patient_id"].values
        folds = []
        for train_idx, valid_idx in self.kf.split(X, y, groups):
            train_df = df.loc[train_idx]
            valid_df = df.loc[valid_idx]
            folds.append((train_df, valid_df))
        self.folds = folds
        return folds

    def save(self, path: str) -> None:
        for i, (train_df, valid_df) in enumerate(self.folds):
            train_df.to_csv(
                f"{path}/train.{self.seed}.{self.n_splits}fold{i}.csv", index=False
            )
            valid_df.to_csv(
                f"{path}/valid.{self.seed}.{self.n_splits}fold{i}.csv", index=False
            )
