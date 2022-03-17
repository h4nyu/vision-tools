from __future__ import annotations

import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn.model_selection import StratifiedKFold
from toolz.curried import filter, map, pipe, topk
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
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
    annotations: pd.DataFrame, n_splits: int, fold_id: 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=n_splits)
    x = list(range(len(annotations)))
    y = annotations["cost"].values // 1000
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        if i == fold_id:
            return annotations.iloc[train_index], annotations.iloc[test_index]


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
        annotations: pd.DataFrame,
        transform: Any,
        image_dir: str,
    ) -> None:
        self.rows = annotations
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
    def __init__(self, dataset: DrawingDataset, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])


def train(cfg: dict, annotations: pd.DataFrame) -> None:
    lt = LtConvNext(cfg)
    trainer = Trainer()


def eda(annotations: pd.DataFrame) -> None:
    ...
