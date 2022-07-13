from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from logging import FileHandler, Formatter, StreamHandler
from typing import Any, Callable, Optional

import albumentations as A
import cv2
import numpy as np
import optuna
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchmetrics
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_metric_learning.losses import ArcFaceLoss
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(Formatter("%(message)s"))
file_handler = FileHandler(f"app.log")
logging.basicConfig(level=logging.NOTSET, handlers=[stream_handler, file_handler])


@dataclass
class Config:
    name: str
    seed: int
    fold: int
    model_name: str
    lr: float
    batch_size: int
    image_size: int
    embedding_size: int
    arcface_scale: float
    arcface_margin: float
    epochs: int
    patience: int
    n_neighbors: int

    scale_limit: float = 0.3
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: float = 0.2
    sat_shift_limit: float = 0.2
    val_shift_limit: float = 0.2
    monitor_mode: str = "min"
    monitor_target: str = "val_loss"
    device: str = "cuda"
    num_workers: int = 11
    num_categories: int = 2
    num_colors: int = 11
    num_model_nos: int = 122
    kfold: int = 5
    use_amp: bool = True
    total_steps: int = 100_00
    topk: int = 10

    resume: str = "score"
    checkpoint_dir: str = "models"

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
        return Config(**obj)

    @property
    def checkpoint_filename(self) -> str:
        return f"{self.name}.{self.fold}.{self.model_name}.ac-{self.arcface_scale:.3f}.am-{self.arcface_margin:.3f}.emb-{self.embedding_size}.img-{self.image_size}.bs-{self.batch_size}"

    @property
    def last_checkpoint_filename(self) -> str:
        return f"{self.checkpoint_filename}-last"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.checkpoint_filename + ".ckpt")

    @property
    def last_checkpoint_path(self) -> str:
        return os.path.join(
            self.checkpoint_dir, self.last_checkpoint_filename + ".ckpt"
        )


class MAPKMetric:
    def __init__(self, topk: int) -> None:
        self.topk = topk
        self.update_count = 0
        self.score_sum = 0.0

    def update(self, preds: torch.Tensor, gts: torch.Tensor) -> None:
        """average_precision"""
        num_positives = len(gts)
        pred_count = 0
        dtc_count = 0
        score = 0.0
        gts_set = set(gts.tolist())
        for pred in preds.tolist():
            pred_count += 1
            if pred in gts_set:
                dtc_count += 1
                score += dtc_count / pred_count
                gts_set.remove(pred)
            if len(gts) == 0:
                break
        score /= min(num_positives, self.topk)
        self.update_count += 1
        self.score_sum += score

    def compute(self) -> float:
        return self.score_sum / max(self.update_count, 1)

    def reset(
        self,
    ) -> None:
        self.update_count = 0
        self.score_sum = 0


def preprocess(
    image_dir: str, meta_path: Optional[str] = None, meta: Optional[dict] = None
) -> dict:
    rows = []
    if meta_path is not None and meta is None:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    if meta is None:
        return {}
    for key in meta:
        filenames = os.listdir(os.path.join(image_dir, key))
        for filename in filenames:
            row = {}
            row["model_no"] = key
            row["category"] = meta[key]["category"]
            row["color"] = meta[key]["color"]
            row["combined_label"] = f"{row['category']}_{row['color']}"
            row["image_path"] = os.path.join(image_dir, key, filename)
            rows.append(row)

    model_no_le = preprocessing.LabelEncoder()
    model_no_le.fit([row["model_no"] for row in rows])
    model_no_labels = model_no_le.transform([row["model_no"] for row in rows])

    category_le = preprocessing.LabelEncoder()
    category_le.fit([row["category"] for row in rows])
    category_labels = category_le.transform([row["category"] for row in rows])

    color_le = preprocessing.LabelEncoder()
    color_le.fit([row["color"] for row in rows])
    color_labels = color_le.transform([row["color"] for row in rows])

    combined_le = preprocessing.LabelEncoder()
    combined_le.fit([row["combined_label"] for row in rows])
    combined_labels = combined_le.transform([row["combined_label"] for row in rows])
    for i, row in enumerate(rows):
        row["category_label"] = category_labels[i]
        row["color_label"] = color_labels[i]
        row["combined_label"] = combined_labels[i]
        row["model_no_label"] = model_no_labels[i]
    encoders = dict(
        category=category_le,
        color=color_le,
        combined=combined_le,
        model_no=model_no_le,
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
    print("Number of unique part ids:", len(set([row["model_no"] for row in rows])))


TrainTransform = lambda cfg: A.Compose(
    [
        A.LongestMaxSize(max_size=cfg.image_size),
        A.PadIfNeeded(
            min_height=cfg.image_size,
            min_width=cfg.image_size,
            border_mode=cv2.BORDER_REPLICATE,
        ),
        A.ShiftScaleRotate(scale_limit=cfg.scale_limit, rotate_limit=180, p=0.5),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=cfg.hue_shift_limit,
                    sat_shift_limit=cfg.sat_shift_limit,
                    val_shift_limit=cfg.val_shift_limit,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.brightness_limit,
                    p=0.5,
                ),
            ],
            p=0.9,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Rotate(p=0.5),
        ToTensorV2(),
    ],
)

InferenceTransform = lambda cfg: A.Compose(
    [
        A.LongestMaxSize(max_size=cfg.image_size),
        A.PadIfNeeded(
            min_height=cfg.image_size,
            min_width=cfg.image_size,
            border_mode=cv2.BORDER_REPLICATE,
        ),
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

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = io.imread(row["image_path"])
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        category_label = torch.tensor(row["category_label"] or 0)
        color_label = torch.tensor(row["color_label"] or 0)
        model_no_label = torch.tensor(row["model_no_label"] or 0)
        sample = dict(
            image=image,
            category_label=category_label,
            color_label=color_label,
            model_no_label=model_no_label,
        )
        return dict(sample=sample, row=row)


def preview_dataset(cfg: Config, rows: list[dict], path: str) -> None:
    dataset = TanachoDataset(
        rows=rows,
        transform=TrainTransform(cfg),
    )
    grid = make_grid([dataset[i]["sample"]["image"] for i in range(10)])
    save_image(grid, path)


def kfold(cfg: Config, rows: list[dict]) -> list[dict]:
    """fold by model_no"""
    skf = StratifiedKFold(n_splits=cfg.kfold)
    y = [row["model_no"] for row in rows]
    X = range(len(rows))
    folds = []
    for fold_, (train_, valid_) in enumerate(skf.split(X=X, y=y)):
        folds.append(
            {
                "train": [rows[i] for i in train_],
                "valid": [rows[i] for i in valid_],
            }
        )
    return folds


# TODO: preview folds
def check_folds(rows: list[dict], folds: list[dict]) -> None:
    for i, fold in enumerate(folds):
        train_rows = fold["train"]
        valid_rows = fold["valid"]


class Net(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(name, pretrained)
        self.backbone.reset_classifier(0)
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.backbone.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.backbone.num_features, embedding_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone.forward_features(x)
        embeddings = self.head(feats)
        return F.normalize(embeddings)


class LitModelNoNet(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = Net(
            name=cfg.model_name,
            embedding_size=cfg.embedding_size,
        )
        self.arcface = ArcFaceLoss(
            num_classes=cfg.num_model_nos,
            embedding_size=cfg.embedding_size,
            scale=cfg.arcface_scale,
            margin=cfg.arcface_margin,
        )
        self.save_hyperparameters()

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        sample, batch_idx = args
        batch = sample["sample"]
        embeddings = self.net(batch["image"])
        loss = self.arcface(embeddings, batch["model_no_label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        sample, batch_idx = args
        batch = sample["sample"]
        return self.net(batch["image"]), sample["row"]

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        sample, batch_idx = args
        batch = sample["sample"]
        embeddings = self.net(batch["image"])
        loss = self.arcface(embeddings, batch["model_no_label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def lr_scheduler_step(
        self, scheduler: Any, optimizer_idx: Any, metric: Any
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.lr,
        )
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.cfg.epochs,
            lr_min=1e-6,
            warmup_t=3,
            warmup_lr_init=1e-5,
            warmup_prefix=True,
        )
        return [optimizer], [scheduler]


def train(cfg: Config, fold: dict) -> LitModelNoNet:
    print(cfg)
    train_dataset = TanachoDataset(
        rows=fold["train"],
        transform=TrainTransform(cfg),
    )
    valid_dataset = TanachoDataset(
        rows=fold["valid"],
        transform=InferenceTransform(cfg),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    trainer = pl.Trainer(
        deterministic=True,
        precision=16,
        gpus=1,
        max_epochs=-1,
        accelerator="gpu",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor=cfg.monitor_target,
                mode=cfg.monitor_mode,
                dirpath=cfg.checkpoint_dir,
                filename=cfg.checkpoint_filename,
                auto_insert_metric_name=False,
            ),
            pl.callbacks.EarlyStopping(
                monitor=cfg.monitor_target, mode=cfg.monitor_mode, patience=cfg.patience
            ),
        ],
    )
    model = LitModelNoNet(cfg)
    trainer.fit(
        model,
        train_loader,
        valid_loader,
    )
    trainer.save_checkpoint(cfg.last_checkpoint_path)
    return model


def evaluate(cfg: Config, fold: dict, model: Optional[LitModelNoNet] = None) -> float:
    val_dataset = TanachoDataset(
        rows=fold["valid"],
        transform=InferenceTransform(cfg),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    net = model or LitModelNoNet.load_from_checkpoint(cfg.last_checkpoint_path)

    registry = Registry(
        rows=fold["train"],
        model=net,
        cfg=cfg,
    )
    registry.create_index()

    # I don't know how to get predictions with multiple gpus
    trainer = pl.Trainer(
        deterministic=True,
        gpus=1,
        max_epochs=-1,
        precision=16,
        accelerator="gpu",
    )
    val_preds: Any = trainer.predict(net, val_loader)
    val_embeds = np.empty([0, cfg.embedding_size])
    val_labels = np.empty(0)

    metric = MAPKMetric(topk=cfg.topk)

    for embeds, row in val_preds:
        labels = row["model_no_label"]
        y = registry.filter_id(embeds)
        for preds, label in zip(torch.tensor(y), labels.unsqueeze(1)):
            metric.update(preds, label)
    score = metric.compute()
    print(f"score={score}")
    return score


class Search:
    def __init__(self, cfg: Config, fold: dict, n_trials: int) -> None:
        self.cfg = cfg
        self.fold = fold
        self.n_trials = n_trials

    def objective(self, trial: optuna.trial.Trial) -> float:
        arcface_scale = trial.suggest_float("arcface_scale", 7.0, 20.0)
        arcface_margin = trial.suggest_float("arcface_margin", 0.0, 5.0)
        embedding_size = trial.suggest_categorical("embedding_size", [768, 1024, 2048])
        image_size = trial.suggest_categorical("image_size", [256, 380, 480, 512])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32])

        # brightness_limit = trial.suggest_float("brightness_limit", 0.0, 0.4)
        # contrast_limit = trial.suggest_float("contrast_limit", 0.0, 0.3)
        # hue_shift_limit = trial.suggest_float("hue_shift_limit", 0.0, 0.3)
        # sat_shift_limit = trial.suggest_float("sat_shift_limit", 0.0, 0.3)
        # val_shift_limit = trial.suggest_float("val_shift_limit", 0.0, 0.3)
        # scale_limit = trial.suggest_float("scale_limit", 0.0, 0.4)

        cfg = Config(
            **{
                **asdict(self.cfg),
                **dict(
                    arcface_scale=arcface_scale,
                    arcface_margin=arcface_margin,
                    embedding_size=embedding_size,
                    image_size=image_size,
                    batch_size=batch_size,
                    # brightness_limit=brightness_limit,
                    # contrast_limit=contrast_limit,
                    # hue_shift_limit=hue_shift_limit,
                    # sat_shift_limit=sat_shift_limit,
                    # val_shift_limit=val_shift_limit,
                    # scale_limit=scale_limit,
                ),
            }
        )
        model = train(cfg=cfg, fold=self.fold)
        score = evaluate(cfg=cfg, fold=self.fold, model=model)
        return score

    def __call__(self) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, self.n_trials)
        print(study.best_value, study.best_params)


class Registry:
    def __init__(
        self,
        model: LitModelNoNet,
        cfg: Config,
        rows: list[dict],
    ) -> None:
        self.neigh = NearestNeighbors(
            n_neighbors=cfg.n_neighbors,
            metric="cosine",
        )
        self.rows = rows
        self.cfg = cfg
        self.n_neighbors = cfg.n_neighbors
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=cfg.image_size),
                A.PadIfNeeded(
                    min_height=cfg.image_size,
                    min_width=cfg.image_size,
                    border_mode=cv2.BORDER_REPLICATE,
                ),
                ToTensorV2(),
            ],
        )
        self.hflip_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=cfg.image_size),
                A.PadIfNeeded(
                    min_height=cfg.image_size,
                    min_width=cfg.image_size,
                    border_mode=cv2.BORDER_REPLICATE,
                ),
                A.HorizontalFlip(p=1.0),
                ToTensorV2(),
            ],
        )
        self.vflip_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=cfg.image_size),
                A.PadIfNeeded(
                    min_height=cfg.image_size,
                    min_width=cfg.image_size,
                    border_mode=cv2.BORDER_REPLICATE,
                ),
                A.VerticalFlip(p=1.0),
                ToTensorV2(),
            ],
        )
        self.model = LitModelNoNet.load_from_checkpoint(cfg.checkpoint_path)
        self.transforms = [
            self.transform,
            self.vflip_transform,
            self.hflip_transform,
        ]
        self.all_labels = np.empty(0)

    def get_dataloader(self, transform: Any) -> DataLoader:
        dataset = TanachoDataset(
            rows=self.rows,
            transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def create_index(self) -> None:
        trainer = pl.Trainer(
            deterministic=True,
            gpus=1,
            max_epochs=-1,
            precision=16,
            accelerator="gpu",
        )

        all_embeds = np.empty([0, self.cfg.embedding_size])
        all_labels = np.empty(0)
        for transform in self.transforms:
            dataloader = self.get_dataloader(self.transform)
            preds: Any = trainer.predict(self.model, dataloader)
            for embeds, row in preds:
                labels = row["model_no_label"]
                all_embeds = np.vstack((all_embeds, embeds.cpu().numpy()))
                all_labels = np.hstack((all_labels, labels.cpu().numpy()))
        self.all_labels = all_labels
        self.neigh.fit(all_embeds, all_labels)

    def filter_id(self, embeddings: Tensor) -> list[int]:
        x = embeddings.cpu().numpy()
        y = self.neigh.kneighbors(
            x, n_neighbors=self.n_neighbors, return_distance=False
        )
        res = np.empty([0, self.n_neighbors])
        for arr in self.all_labels[y]:
            u, ind = np.unique(arr, return_index=True)
            u = u[np.argsort(ind)]
            length = len(u)
            other = np.repeat(u[-1], self.n_neighbors - length)
            u = np.concatenate([u, other])
            res = np.vstack([res, u])
        return res.tolist()


class ScoringService(object):
    model: Any
    reference: Any
    reference_meta: Any
    registry: Registry

    @classmethod
    def get_model(
        cls, model_path: str, reference_path: str, reference_meta_path: str
    ) -> bool:
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        try:
            cls.model = None
            cls.reference = os.listdir(reference_path)
            with open(reference_meta_path) as f:
                cls.reference_meta = json.load(f)

            return True
        except:
            return False

    @classmethod
    def predict(cls, input: str) -> dict:
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        image = io.imread(input)
        sample_name = os.path.basename(input).split(".")[0]

        # make prediction
        prediction = cls.reference[:10]

        # make output
        output = {sample_name: prediction}

        return output
