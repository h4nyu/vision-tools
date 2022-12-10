from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import optuna
import pandas as pd
import timm
import toolz
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def seed_everything(seed: int = 3801) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if ctp + cfp == 0:
        return 0.0

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


class PFBetaMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output: List[Tensor] = []
        self.target: List[Tensor] = []
        self.func: Callable = pfbeta

    def accumulate(self, output: Tensor, target: Tensor) -> None:
        self.output.append(output)
        self.target.append(target)

    def __call__(self) -> float:
        output = torch.cat(self.output)
        target = torch.cat(self.target)
        return self.func(target, output)

    def reset(self) -> None:
        self.output = []
        self.target = []


@dataclass
class Config:
    name: str
    image_size: int = 256
    in_channels: int = 1
    batch_size: int = 48
    lr: float = 1e-3
    optimizer: str = "Adam"
    epochs: int = 100
    seed: int = 42
    n_splits: int = 5
    model_name: str = "tf_efficientnet_b3_ns"
    pretrained: bool = True
    fold: int = 0
    data_path: str = "/store"
    hflip: float = 0.5
    vflip: float = 0.5
    scale_limit: float = 0.3
    rotate_limit: float = 15
    border_mode: int = 0

    @property
    def train_csv(self) -> str:
        return f"{self.data_path}/train.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def valid_csv(self) -> str:
        return f"{self.data_path}/valid.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def image_dir(self) -> str:
        return f"{self.data_path}/rsna-breast-cancer-{self.image_size}-pngs"

    @property
    def log_dir(self) -> str:
        return f"{self.data_path}/logs/{self.name}-{self.seed}-{self.n_splits}fold{self.fold}"

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
        return Config(**obj)


class RdcdPngDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
    ) -> None:
        self.df = df
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = f"{self.image_dir}/{row['patient_id']}_{row['image_id']}.png"
        image = io.imread(image_path)
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        target = torch.tensor([row["cancer"]]) if row["cancer"] is not None else None
        sample = dict(
            image=image,
            target=target,
            patient_id=row["patient_id"],
            image_id=row["image_id"],
        )
        return sample


TrainTransform = lambda cfg: A.Compose(
    [
        A.HorizontalFlip(p=cfg.hflip),
        # A.VerticalFlip(p=cfg.vflip),
        A.ShiftScaleRotate(
            scale_limit=cfg.scale_limit,
            rotate_limit=cfg.rotate_limit,
            p=0.5,
            border_mode=cfg.border_mode,
        ),
        ToTensorV2(),
    ],
)

Transform = lambda cfg: A.Compose(
    [
        ToTensorV2(),
    ],
)


class RdcdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
    ) -> None:
        self.df = df
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
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
        self.folds: list[pd.Dataframe] = []

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


class BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Union[RdcdPngDataset, RdcdDataset],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.df = dataset.df
        self.n_classes = 2
        self.shuffle = shuffle

    def __len__(self) -> int:
        non_cancer_idx = self.df[self.df["cancer"] == 0]
        return len(non_cancer_idx) // (self.batch_size // 2)

    # hard sampleを作るために、どうすればよいか？
    # 同じ年齢帯,　同じ性別の人をまとめて、それぞれのバッチに入れる?
    def __iter__(self) -> Iterator:
        df = self.dataset.df
        cancer_idx = df[df["cancer"] == 1].index.values
        non_cancer_idx = df[df["cancer"] == 0].index.values
        over_sample_idx = cancer_idx.repeat(len(non_cancer_idx) // len(cancer_idx) + 1)
        non_cancer_batch_size = self.batch_size // 2
        cancer_batch_size = self.batch_size - non_cancer_batch_size
        if self.shuffle:
            np.random.shuffle(over_sample_idx)
            np.random.shuffle(non_cancer_idx)
        for i in range(len(self)):
            non_cancer_batch = non_cancer_idx[
                i * non_cancer_batch_size : (i + 1) * non_cancer_batch_size
            ]
            cancer_batch = over_sample_idx[
                i * cancer_batch_size : (i + 1) * cancer_batch_size
            ]
            batch = np.concatenate([non_cancer_batch, cancer_batch])
            yield batch


class Model(nn.Module):
    def __init__(self, name: str, in_channels: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            name, pretrained=pretrained, in_chans=in_channels, num_classes=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return x


class Train:
    def __init__(
        self,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.net = Model(
            name=cfg.model_name,
            in_channels=cfg.in_channels,
            pretrained=cfg.pretrained,
        )
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=cfg.log_dir)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self) -> nn.Module:
        return self.net.to(self.device)

    @property
    def num_workers(self) -> int:
        return os.cpu_count() or 0

    def train_one_epoch(
        self, train_loader: DataLoader, optimizer: Any, criterion: Any
    ) -> Tuple[float, float]:
        metric = PFBetaMetric()
        self.model.train()
        epoch_loss = 0.0
        epoch_score = 0.0
        for batch in tqdm(train_loader):
            image = batch["image"].to(self.device)
            target = batch["target"].to(self.device)
            optimizer.zero_grad()
            with autocast():
                output = self.model(image)
                loss = criterion(output, target.float())
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            metric.accumulate(output.sigmoid(), target)
        epoch_loss /= len(train_loader)
        epoch_score = metric()
        return epoch_loss, epoch_score

    def valid_one_epoch(
        self, valid_loader: DataLoader, criterion: Any
    ) -> Tuple[float, float]:
        metric = PFBetaMetric()
        self.model.eval()
        epoch_loss = 0.0
        epoch_score = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                image = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                output = self.model(image)
                loss = criterion(output, target.float())
                epoch_loss += loss.item()
                metric.accumulate(output.sigmoid(), target)
        epoch_loss /= len(valid_loader)
        epoch_score = metric()
        return epoch_loss, epoch_score

    def __call__(self, limit: Optional[int] = None) -> None:
        cfg = self.cfg
        train_df = pd.read_csv(cfg.train_csv)
        if limit is not None:
            train_df = train_df[:limit]
        valid_df = pd.read_csv(cfg.valid_csv)
        if limit is not None:
            valid_df = valid_df[:limit]
        train_dataset = RdcdPngDataset(
            df=train_df,
            transform=TrainTransform(cfg),
            image_dir=cfg.image_dir,
        )
        valid_dataset = RdcdPngDataset(
            df=valid_df,
            transform=Transform(cfg),
            image_dir=cfg.image_dir,
        )
        batch_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=cfg.batch_size,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        criterion = nn.BCEWithLogitsLoss()
        best_score = 0
        for epoch in range(cfg.epochs):
            train_loss, train_score = self.train_one_epoch(
                train_loader, optimizer, criterion
            )
            valid_loss, valid_score = self.valid_one_epoch(valid_loader, criterion)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/score", train_score, epoch)
            self.writer.add_scalar("valid/loss", valid_loss, epoch)
            self.writer.add_scalar("valid/score", valid_score, epoch)
        #     valid_loss, valid_score = self.valid_one_epoch(valid_loader, criterion)
        #     if valid_score > best_score:
        #         best_score = valid_score
        #         torch.save(self.model.state_dict(), cfg.model_path)
        #     print(
        #         f"epoch: {epoch + 1}, train_loss: {train_loss:.4f}, train_score: {train_score:.4f}, valid_loss: {valid_loss:.4f}, valid_score: {valid_score:.4f}"
        #     )
