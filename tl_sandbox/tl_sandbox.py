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
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        row = self.rows[idx]
        image_path = os.path.join(self.image_dir, row["image_file"])
        im = PIL.Image.open(image_path)
        if im.mode == "L":
            im = im.convert("RGB")
        img_arr = np.array(im)
        transformed = self.transform(
            image=img_arr,
            label=row["label"] or 0,
        )
        image = (transformed["image"] / 255).float()
        label = torch.tensor(transformed["label"])
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
    def __init__(self, cfg: dict) -> None:
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


def train(cfg: dict, annotations: pd.DataFrame) -> None:
    lt = LtConvNext(cfg)


def eda(annotations: pd.DataFrame) -> None:
    ...
