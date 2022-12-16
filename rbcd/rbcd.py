from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from logging import Logger, getLogger
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import albumentations as A
import cv2
import dicomsdl
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def seed_everything(seed: int = 3801) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pfbeta(labels: Any, predictions: Any, beta: float = 1.0) -> float:
    y_true_count = 0
    tp = 0
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


class Meter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output: List[Tensor] = []
        self.target: List[Tensor] = []

    def accumulate(self, output: Tensor, target: Tensor) -> None:
        self.output.append(output)
        self.target.append(target)

    def __call__(self) -> Tuple[Tensor, Tensor]:
        return (
            torch.cat(self.output),
            torch.cat(self.target),
        )

    def reset(self) -> None:
        self.output = []
        self.target = []


def find_best_threshold(
    target: Tensor,
    output: Tensor,
    step: float = 0.02,
    start: float = 0.45,
    end: float = 0.7,
    logger: Optional[Logger] = None,
) -> Tuple[float, float]:
    best_thresold = 0.0
    best_score = 0.0
    logger = logger or getLogger(__name__)
    for thresold in np.arange(start, end, step):
        score = pfbeta(target, (output > thresold).float(), beta=1)
        logger.info(f"thresold: {thresold:.3f}, score: {score:.3f}")
        if score > best_score:
            best_thresold = thresold
            best_score = score
    return float(best_thresold), float(best_score)


@dataclass
class InferenceConfig:
    data_path: str = "/store"
    models_dir: str = "inference_models"
    configs_dir: str = "configs"
    batch_size: int = 32
    config_paths: List[str] = field(default_factory=list)

    @property
    def configs(self) -> List[Config]:
        overwrite = dict(models_dir=self.models_dir)
        return [
            Config.load(f"{self.configs_dir}/{path}", overwrite)
            for path in self.config_paths
        ]

    @property
    def test_csv(self) -> str:
        return f"{self.data_path}/test.csv"

    @property
    def image_dir(self) -> str:
        return f"{self.data_path}/test_images"

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
        return InferenceConfig(**obj)


@dataclass
class Config:
    name: str
    image_size: int = 256
    in_channels: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "Adam"
    epochs: int = 16
    seed: int = 42
    n_splits: int = 5
    model_name: str = "tf_efficientnet_b2_ns"
    pretrained: bool = True
    accumulation_steps: int = 1
    fold: int = 0
    data_path: str = "/store"
    models_dir: str = "models"
    hflip: float = 0.5
    vflip: float = 0.5
    scale_limit: float = 0.3
    rotate_limit: float = 15
    border_mode: int = 0
    valid_epochs: int = 10
    ratio: float = 1.0
    sampling: str = "over"  # deprecated
    device: str = "cuda"

    search_limit: Optional[int] = None

    @property
    def train_csv(self) -> str:
        return f"{self.data_path}/train.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def valid_csv(self) -> str:
        return f"{self.data_path}/valid.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def image_dir(self) -> str:
        return f"{self.data_path}/images_as_pngs_{self.image_size}/train_images_processed_{self.image_size}"

    @property
    def log_dir(self) -> str:
        return f"{self.data_path}/logs/{self.name}-{self.seed}-{self.n_splits}fold{self.fold}"

    @property
    def model_path(self) -> str:
        return f"{self.models_dir}/{self.name}-{self.seed}-{self.n_splits}fold{self.fold}.pth"

    def __str__(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False)

    @classmethod
    def load(cls, path: str, overwrite: Optional[dict] = None) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
        if overwrite:
            obj.update(overwrite)
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
        image_path = f"{self.image_dir}/{row['patient_id']}/{row['image_id']}.png"
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
        A.VerticalFlip(p=cfg.vflip),
        A.RandomRotate90(),
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


class RdcdDicomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
        image_size: int = 2048,
    ) -> None:
        self.df = df
        self.transform = transform
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    # def convert_dicom_to_j2k(self, row: dict):
    #     patient = row["patient_id"]
    #     image = row["image_id"]
    #     file_path = f"{self.image_dir}/{patient}/{image}.dcm"
    #     dcmfile = pydicom.dcmread(file_path)
    #     if dcmfile.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.90":
    #         with open(file_path, "rb") as fp:
    #             raw = DicomBytesIO(fp.read())
    #             ds = pydicom.dcmread(raw)
    #         offset = ds.PixelData.find(
    #             b"\x00\x00\x00\x0C"
    #         )  # <---- the jpeg2000 header info we're looking for
    #         hackedbitstream = bytearray()
    #         hackedbitstream.extend(ds.PixelData[offset:])
    #         with open(f"../working/{patient}_{image}.jp2", "wb") as binary_file:
    #             binary_file.write(hackedbitstream)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = f"{self.image_dir}/{row['patient_id']}/{row['image_id']}.dcm"
        dicom = dicomsdl.open(image_path)
        img = dicom.pixelData()
        img = (img - img.min()) / (img.max() - img.min())
        if dicom.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
            img = 1 - img
        image = (img * 255).astype(np.uint8)
        image = cv2.resize(image, (self.image_size, self.image_size))
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        target = torch.tensor([row["cancer"]]) if row["cancer"] is not None else None
        return dict(
            image=image,
            target=target,
            patient_id=row["patient_id"],
            image_id=row["image_id"],
        )


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


class OverBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Union[RdcdPngDataset, RdcdDicomDataset],
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
        if self.shuffle:
            np.random.shuffle(non_cancer_idx)

        over_samples = []
        for _ in range(len(non_cancer_idx) // len(cancer_idx) + 1):
            acc = cancer_idx.copy()
            if self.shuffle:
                np.random.shuffle(acc)
            over_samples.append(acc)

        over_sample_idx = np.concatenate(over_samples)[: len(non_cancer_idx)]
        non_cancer_batch_size = self.batch_size // 2
        cancer_batch_size = self.batch_size - non_cancer_batch_size
        for i in range(len(self)):
            non_cancer_batch = non_cancer_idx[
                i * non_cancer_batch_size : (i + 1) * non_cancer_batch_size
            ]
            cancer_batch = over_sample_idx[
                i * cancer_batch_size : (i + 1) * cancer_batch_size
            ]
            batch = np.concatenate([non_cancer_batch, cancer_batch])
            yield batch


class UnderBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Union[RdcdPngDataset, RdcdDicomDataset],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.df = dataset.df
        self.n_classes = 2
        self.shuffle = shuffle

    def __len__(self) -> int:
        cancer_idx = self.df[self.df["cancer"] == 1]
        return len(cancer_idx) // (self.batch_size // 2)

    def __iter__(self) -> Iterator:
        df = self.dataset.df
        cancer_idx = df[df["cancer"] == 1].index.values
        non_cancer_idx = df[df["cancer"] == 0].index.values
        non_cancer_batch_size = self.batch_size // 2
        cancer_batch_size = self.batch_size - non_cancer_batch_size
        if self.shuffle:
            np.random.shuffle(cancer_idx)
            np.random.shuffle(non_cancer_idx)
        for i in range(len(self)):
            non_cancer_batch = non_cancer_idx[
                i * non_cancer_batch_size : (i + 1) * non_cancer_batch_size
            ]
            cancer_batch = cancer_idx[
                i * cancer_batch_size : (i + 1) * cancer_batch_size
            ]
            batch = np.concatenate([non_cancer_batch, cancer_batch])
            yield batch


class OverSampler(Sampler):
    def __init__(
        self,
        dataset: Union[RdcdPngDataset, RdcdDicomDataset],
        shuffle: bool = True,
        ratio: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.df = dataset.df
        self.pos_idx = self.df[self.df["cancer"] == 1].index.values
        self.neg_idx = self.df[self.df["cancer"] == 0].index.values
        self.shuffle = shuffle
        self.ratio = ratio
        self.neg_len = len(self.neg_idx)
        self.pos_len = len(self.pos_idx)
        self.over_len = int(self.neg_len * self.ratio)
        self.length = self.neg_len + self.over_len

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator:
        neg_idx = self.neg_idx.copy()
        pos_idx = np.repeat(self.pos_idx, self.over_len // self.pos_len + 1)[
            : self.over_len
        ]
        idx = np.concatenate([neg_idx, pos_idx])
        if self.shuffle:
            np.random.shuffle(idx)
        return iter(idx)


class Model(nn.Module):
    def __init__(self, name: str, in_channels: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            name, pretrained=pretrained, in_chans=in_channels, num_classes=1
        )

    @classmethod
    def load(cls, cfg: Config) -> "Model":
        model = cls(name=cfg.model_name, in_channels=cfg.in_channels)
        state_dict = torch.load(cfg.model_path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return x


class Train:
    def __init__(
        self,
        cfg: Config,
        logger: Optional[Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.net = Model(
            name=cfg.model_name,
            in_channels=cfg.in_channels,
            pretrained=cfg.pretrained,
        )
        self.logger = logger or getLogger(__name__)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.iteration = 0

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
        self,
        train_loader: DataLoader,
        optimizer: Any,
        criterion: Any,
    ) -> Tuple[float, float]:
        self.model.train()
        meter = Meter()
        epoch_loss = 0.0
        epoch_score = 0.0
        for batch in tqdm(train_loader):
            self.iteration += 1
            image = batch["image"].to(self.device)
            target = batch["target"].to(self.device)
            optimizer.zero_grad()
            with autocast():
                output = self.model(image)
                loss = criterion(output, target.float()) / self.cfg.accumulation_steps
            self.scaler.scale(loss).backward()
            if self.iteration % self.cfg.accumulation_steps == 0:
                self.scaler.unscale_(optimizer)
                self.scaler.step(optimizer)
                self.scaler.update()
            epoch_loss += loss.item()
            meter.accumulate(output.sigmoid(), target)
        epoch_loss /= len(train_loader)
        output, target = meter()
        epoch_score = pfbeta(
            target,
            output,
        )
        return float(epoch_loss), float(epoch_score)

    def __call__(self, limit: Optional[int] = None) -> float:
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
        sampler = OverSampler(train_dataset, shuffle=True, ratio=cfg.ratio)
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=cfg.batch_size,
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
        best_score = 0.0
        validate = Validate(self.cfg)
        for epoch in range(cfg.epochs):
            train_loss, train_score = self.train_one_epoch(
                train_loader, optimizer, criterion
            )
            self.writer.add_scalar("train/loss", train_loss, self.iteration)
            self.writer.add_scalar("train/score", train_score, self.iteration)
            if epoch % cfg.valid_epochs == 0:
                (
                    valid_loss,
                    valid_score,
                    valid_auc,
                    valid_binary_score,
                    valid_threshold,
                ) = validate(self.model)
                self.writer.add_scalar("valid/loss", valid_loss, self.iteration)
                self.writer.add_scalar("valid/score", valid_score, self.iteration)
                self.writer.add_scalar("valid/auc", valid_auc, self.iteration)
                self.writer.add_scalar(
                    "valid/valid_binary_score", valid_binary_score, self.iteration
                )
                self.writer.add_scalar(
                    "valid/valid_threshold", valid_threshold, self.iteration
                )
                self.writer.flush()

                if valid_score > best_score:
                    best_score = valid_score
                    torch.save(self.model.state_dict(), cfg.model_path)
                self.logger.info(
                    f"epoch: {epoch + 1}, iteration: {self.iteration}, train_loss: {train_loss:.4f}, train_score: {train_score:.4f}, valid_loss: {valid_loss:.4f}, valid_score: {valid_score:.4f}, valid_auc: {valid_auc:.4f}, valid_binary_score: {valid_binary_score:.4f}, valid_threshold: {valid_threshold:.4f}"
                )
        return valid_binary_score


class Validate:
    def __init__(
        self,
        cfg: Config,
        logger: Optional[Logger] = None,
    ) -> None:
        valid_df = pd.read_csv(cfg.valid_csv)
        valid_dataset = RdcdPngDataset(
            df=valid_df,
            transform=Transform(cfg),
            image_dir=cfg.image_dir,
        )
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=cfg.batch_size,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.logger = logger or getLogger(cfg.name)

    @property
    def num_workers(self) -> int:
        return os.cpu_count() or 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(
        self, net: nn.Module, enable_find_threshold: bool = False
    ) -> Tuple[float, float, float, float, float]:
        meter = Meter()
        net.eval()
        epoch_loss = 0.0
        epoch_score = 0.0
        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                image = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                output = net(image)
                loss = self.criterion(output, target.float())
                epoch_loss += loss.item()
                meter.accumulate(output.sigmoid().flatten(), target.flatten())
        epoch_loss /= len(self.valid_loader)
        output, target = meter()
        epoch_score = pfbeta(
            target.cpu().numpy(),
            output.cpu().numpy(),
        )
        epoch_auc = roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
        best_threshold, binary_score = find_best_threshold(target, output)
        return (
            float(epoch_loss),
            float(epoch_score),
            float(epoch_auc),
            float(binary_score),
            float(best_threshold),
        )


class Search:
    def __init__(
        self,
        cfg: Config,
        n_trials: int,
        limit: Optional[int] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.n_trials = n_trials
        self.limit = limit
        self.logger = logger or getLogger(cfg.name)

    def objective(self, trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1.5e-3)
        ratio = trial.suggest_float("ratio", 0.1, 1.0)
        accumulation_steps = trial.suggest_int("accumulation_steps", 1, 4)
        cfg = Config(
            **{
                **asdict(self.cfg),
                **dict(
                    name=f"{self.cfg.name}_ratio{ratio}_lr{lr}_accumulation_steps{accumulation_steps}",
                    lr=lr,
                    ratio=ratio,
                    accumulation_steps=accumulation_steps,
                ),
            }
        )
        self.logger.info(f"trial: {trial.number}, cfg: {cfg}")
        train = Train(cfg, logger=self.logger)
        validate = Validate(cfg, logger=self.logger)
        train(self.limit)
        model = Model.load(cfg).to(cfg.device)
        loss, score, auc, binary_score, thr = validate(model)
        self.logger.info(
            f"trial: {trial.number}, cfg: {cfg} loss: {loss:.4f}, score: {score:.4f}, auc: {auc:.4f}, binary_score: {binary_score:.4f}, thr: {thr:.4f}"
        )
        return binary_score

    def __call__(self) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, self.n_trials)
        self.logger.info(
            f"best_params: {study.best_params} best_value: {study.best_value}"
        )


class Inference:
    def __init__(
        self,
        cfg: InferenceConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or getLogger(__name__)

    @property
    def num_workers(self) -> int:
        return os.cpu_count() or 0

    @torch.no_grad()
    def inference(self, cfg: Config, loader: DataLoader) -> np.ndarray:
        model = Model.load(cfg).to(cfg.device)

    def __call__(self) -> None:
        df = pd.read_csv(self.cfg.test_csv)
        dataset = RdcdDicomDataset(
            df=df,
            transform=Transform(self.cfg),
            image_dir=self.cfg.image_dir,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.num_workers,
        )
        for cfg in self.cfg.configs:
            self.logger.info(f"cfg: {cfg}")
            output = self.inference(cfg, loader)
            df[cfg.name] = output
