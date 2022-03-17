from __future__ import annotations

import os
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pandas import DataFrame
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn.model_selection import StratifiedKFold
from toolz.curried import filter, map, pipe, topk
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

Transform = lambda cfg: A.Compose(
    [
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_NEAREST,
        ),
        ToTensorV2(),
    ],
)


def kfold(
    rows: Any,
    n_splits: int,
    fold_id: int,
) -> tuple[Any, Any]:
    skf = StratifiedKFold(n_splits=n_splits)
    x = list(range(len(rows)))
    y = rows["cost"].values // 1000
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        if i == fold_id:
            return rows.iloc[train_index], rows.iloc[test_index]
    raise ValueError("fold_id is out of range")


class Writer(SummaryWriter):
    def __init__(self, cfg: dict) -> None:
        writer_name = pipe(
            cfg.items(),
            map(lambda x: f"{x[0]}-{x[1]}"),
            "_".join,
        )
        super().__init__(
            f"runs/{writer_name}",
            flush_secs=5,
        )


class DrawingDataset(Dataset):
    def __init__(
        self,
        rows: Any,
        transform: Any,
        image_dir: str,
    ) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        row = self.rows.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        im = PIL.Image.open(image_path).convert("RGB")
        if im.mode == "L":
            im = im.convert("RGB")
        img_arr = np.array(im)
        transformed = self.transform(
            image=img_arr,
        )
        image = (transformed["image"] / 255).float()
        label = torch.tensor(row["cost"])
        sample = dict(
            image=image,
            label=label,
        )
        return sample, row


class ConvNeXt(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained, num_classes=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        out = F.normalize(out, p=2, dim=1)
        return out


class LtConvNext(LightningModule):
    def __init__(
        self,
        cfg: dict,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = ConvNeXt(
            name=cfg["model_name"],
            embedding_size=cfg["embedding_size"],
            pretrained=cfg["pretrained"],
        )
        self.loss_fn = ArcFaceLoss(
            num_classes=cfg["num_classes"],
            embedding_size=cfg["embedding_size"],
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"])

    def forward(self, batch: Any) -> Tensor:  # type: ignore
        x, y = batch
        embeddings = self.model(x)
        loss = self.loss_fn(embeddings, y)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:  # type: ignore
        loss = self(batch)
        self.log("train/loss", loss)
        return loss


class LtDrawingDataModule(LightningDataModule):
    def __init__(self, annotations: DataFrame, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.annotations = annotations

    def setup(self, stage: Optional[str] = None) -> None:
        train_rows, val_rows = kfold(
            self.annotations, n_splits=self.cfg["n_splits"], fold_id=self.cfg["fold_id"]
        )
        self.train_set = DrawingDataset(
            rows=train_rows,
            transform=Transform(self.cfg),
            image_dir=self.cfg["image_dir"],
        )
        self.val_set = DrawingDataset(
            rows=val_rows,
            transform=Transform(self.cfg),
            image_dir=self.cfg["image_dir"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.cfg["batch_size"])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.cfg["batch_size"])


def train(cfg: dict, annotations: Any) -> None:
    lt = LtConvNext(cfg)
    trainer = Trainer()


def eda(annotations: Any) -> None:
    ...
