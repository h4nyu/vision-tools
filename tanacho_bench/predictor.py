from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle
import pprint as pp
import random
from dataclasses import asdict, dataclass, field
from logging import FileHandler, Formatter, StreamHandler
from typing import Any, Callable, Iterator, Optional

import albumentations as A
import cv2
import numpy as np
import optuna
import pytorch_lightning as pl
import timm
import toolz
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_metric_learning.losses import ArcFaceLoss, SubCenterArcFaceLoss
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from timm.scheduler import CosineLRScheduler
from toolz.curried import compose, curry, filter, groupby, map, partition, pipe, sorted
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
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
    loss_type: str = "ArcFaceLoss"
    patience: int = 13
    epochs: int = 60
    n_neighbors: int = 200
    max_epochs: int = -1
    warmup_t: int = 5
    border_mode: int = 0
    scale_limit: float = 0.3
    brightness_limit: float = 0.01
    contrast_limit: float = 0.01
    hue_shift_limit: float = 0.2
    sat_shift_limit: float = 0.2
    val_shift_limit: float = 0.2
    rotate_limit: float = 15
    sub_centers: int = 8
    monitor_mode: str = "min"
    monitor_target: str = "val_loss"
    device: str = "cuda"
    num_workers: int = 11
    num_categories: int = 2
    num_colors: int = 11
    num_model_nos: int = 122
    kfold: int = 4
    use_amp: bool = True
    total_steps: int = 100_00
    topk: int = 10
    pretrained: bool = True
    fine: dict = field(default_factory=dict)
    resume: str = "score"
    checkpoint_dir: str = "checkpoints"
    hard_samples: list[str] = field(default_factory=list)
    is_fine: bool = False
    vflip_p: float = 0.5
    hflip_p: float = 0.5
    blur_p: float = 0.0
    train_with_flipped: bool = False
    random_rotate_p: float = 0.5
    accumulate_grad_batches: int = 1
    previous: Optional[dict] = None
    use_scheduler: bool = True
    registry_rot_augmentations: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as file:
            obj = yaml.safe_load(file)
        return Config(**obj)

    @property
    def checkpoint_filename(self) -> str:
        return f"{self.name}.{self.fold}.{self.kfold}.{self.model_name}.{self.loss_type}.ac-{self.arcface_scale:.3f}.am-{self.arcface_margin:.3f}.emb-{self.embedding_size}.img-{self.image_size}.bs-{self.batch_size}.fine-{self.is_fine}"

    @property
    def previous_checkpoint_path(self) -> Optional[str]:
        if self.previous is not None:
            cfg = Config(
                **{
                    **asdict(self),
                    **self.previous,
                }
            )
            return cfg.checkpoint_path
        return None

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.checkpoint_filename + ".ckpt")


@dataclass
class EnsembleConfig:
    config_paths: list[str] = field(default_factory=list)
    configs: list[Config] = field(default_factory=list)
    topk: int = 10

    @classmethod
    def load(cls, path: str) -> EnsembleConfig:
        with open(path) as file:
            obj = yaml.safe_load(file)
        config_paths = obj["config_paths"]
        configs = [Config.load(p) for p in config_paths]
        return EnsembleConfig(configs=configs, config_paths=config_paths)


class BalancedBatchSampler(BatchSampler):
    def __init__(
        self, dataset: TanachoDataset, batch_size: int, categories: int = 2
    ) -> None:
        self.dataset = dataset
        self.rows = dataset.rows
        self.batch_size = batch_size
        self.categorys = categories

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator:
        self.batch_idx = 0
        groups = toolz.groupby(lambda x: x[1]["category"], enumerate(self.rows))
        batches = []
        # print('-----------')
        for group in groups.values():
            random.shuffle(group)
            rows = pipe(
                group,
                sorted(key=lambda x: x[1]["color"]),
                partition(self.batch_size),
                list,
            )
            batches.extend(rows)
        random.shuffle(batches)
        for batch in batches:
            indices = [x[0] for x in batch]
            yield indices


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
) -> list[dict]:
    rows = []
    if meta_path is not None and meta is None:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    if meta is None:
        return []
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
    return rows


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
            border_mode=cfg.border_mode,
        ),
        A.ShiftScaleRotate(
            scale_limit=cfg.scale_limit,
            rotate_limit=cfg.rotate_limit,
            p=0.5,
            border_mode=cfg.border_mode,
        ),
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
                    contrast_limit=cfg.contrast_limit,
                    p=0.5,
                ),
            ],
            p=0.9,
        ),
        A.RandomRotate90(p=cfg.random_rotate_p),
        A.HorizontalFlip(p=cfg.hflip_p),
        A.VerticalFlip(p=cfg.vflip_p),
        ToTensorV2(),
    ],
)

InferenceTransform = lambda cfg: A.Compose(
    [
        A.LongestMaxSize(max_size=cfg.image_size),
        A.PadIfNeeded(
            min_height=cfg.image_size,
            min_width=cfg.image_size,
            border_mode=cfg.border_mode,
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


def compare_sample_pair(
    cfg: Config, reference: dict, target: dict, path: str, nrow: int = 4
) -> None:
    dataset = TanachoDataset(
        rows=[reference, target],
        transform=TrainTransform(cfg),
    )
    samples = [dataset[0]["sample"]["image"] for i in range(nrow)] + [
        dataset[1]["sample"]["image"] for i in range(nrow)
    ]
    grid = make_grid(samples, nrow=nrow)
    save_image(grid, path)


def kfold(cfg: Config, rows: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    """fold by model_no"""
    skf = StratifiedKFold(n_splits=cfg.kfold)
    y = [row["model_no"] for row in rows]
    X = range(len(y))
    folds = []
    for fold_, (train_, valid_) in enumerate(skf.split(X=X, y=y)):
        folds.append(([rows[i] for i in train_], [rows[i] for i in valid_]))
    return folds


def extend_dataset(rows: list[dict]) -> dict:
    meta: dict = {}
    transform_lr = A.Compose(
        [
            A.HorizontalFlip(always_apply=True),
        ]
    )
    transform_td = A.Compose(
        [
            A.VerticalFlip(always_apply=True),
        ]
    )
    for row in tqdm(rows):
        image_path = pathlib.Path(row["image_path"])
        image = io.imread(image_path)
        # LR
        model_no = "lr-inverted" + row["model_no"]
        base_dir = image_path.parents[1]
        model_no_dir = base_dir / pathlib.Path(model_no)
        model_no_dir.mkdir(exist_ok=True)
        new_image_path = (
            base_dir / pathlib.Path(model_no) / pathlib.Path(image_path.name)
        )
        transformed = transform_lr(
            image=image,
        )["image"]
        meta[model_no] = {
            "category": row["category"],
            "color": row["color"],
        }
        io.imsave(new_image_path, transformed)

        # TD
        # model_no = "td-inverted" + row["model_no"]
        # base_dir = image_path.parents[1]
        # model_no_dir = base_dir / pathlib.Path(model_no)
        # model_no_dir.mkdir(exist_ok=True)
        # new_image_path = (
        #     base_dir / pathlib.Path(model_no) / pathlib.Path(image_path.name)
        # )
        # transformed = transform_td(
        #     image=image,
        # )["image"]
        # meta[model_no] = {
        #     "category": row["category"],
        #     "color": row["color"],
        # }
        # io.imsave(new_image_path, transformed)

    return meta


# TODO: preview folds
def check_folds(rows: list[dict], folds: list[dict]) -> None:
    for i, fold in enumerate(folds):
        train_rows = fold["train"]
        valid_rows = fold["valid"]


class Net(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(name, pretrained=pretrained)
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
            pretrained=cfg.pretrained,
        )
        if cfg.loss_type == "SubCenterArcFaceLoss":
            self.arcface = SubCenterArcFaceLoss(
                num_classes=cfg.num_model_nos,
                embedding_size=cfg.embedding_size,
                scale=cfg.arcface_scale,
                margin=cfg.arcface_margin,
                sub_centers=cfg.sub_centers,
            )
        if cfg.loss_type == "ArcFaceLoss":
            self.arcface = ArcFaceLoss(
                num_classes=cfg.num_model_nos,
                embedding_size=cfg.embedding_size,
                scale=cfg.arcface_scale,
                margin=cfg.arcface_margin,
            )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.net(x)

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
        if self.cfg.use_scheduler:
            scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.lr,
        )
        if self.cfg.use_scheduler:
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.cfg.epochs,
                lr_min=1e-6,
                warmup_t=self.cfg.warmup_t,
                warmup_lr_init=1e-5,
                warmup_prefix=True,
            )
            return [optimizer], [scheduler]
        return [optimizer], []


def train(cfg: Config, fold: dict) -> LitModelNoNet:
    pp.pprint(cfg)
    train_rows = fold["train"]
    for sample in cfg.hard_samples:
        for row in fold["valid"]:
            if row["image_path"].endswith(sample):
                train_rows.append(row)

    train_dataset = TanachoDataset(
        rows=fold["train"],
        transform=TrainTransform(cfg),
    )
    valid_dataset = TanachoDataset(
        rows=fold["valid"],
        transform=InferenceTransform(cfg),
    )
    batch_sampler = BalancedBatchSampler(train_dataset, batch_size=cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_sampler=batch_sampler,
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
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=cfg.monitor_target, mode=cfg.monitor_mode, patience=cfg.patience
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=cfg.checkpoint_dir,
                filename=cfg.checkpoint_filename,
                save_last=False,
            ),
        ],
    )
    model = (
        LitModelNoNet.load_from_checkpoint(cfg.previous_checkpoint_path, cfg=cfg)
        if cfg.previous_checkpoint_path
        else LitModelNoNet(cfg)
    )
    trainer.fit(
        model,
        train_loader,
        valid_loader,
    )
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
    net = model or LitModelNoNet.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)

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

        for i, (preds, label) in enumerate(zip(torch.tensor(y), labels.unsqueeze(1))):
            metric.update(preds, label)
            if preds[0] != label:
                image_path = row["image_path"][i]
                print(f"image_path={image_path}")
                expected = registry.label_map[int(label)]
                actual = registry.label_map[int(preds[0])]
                print(f"predicted={[registry.label_map[int(i)] for i in preds]}")
                print(f"expected={expected}, actual={actual}")
                print(f"=================")

    score = metric.compute()
    print(f"score={score}")
    return score


class SearchRegistry:
    def __init__(self, cfg: Config, fold: dict, n_trials: int) -> None:
        self.cfg = cfg
        self.fold = fold
        self.n_trials = n_trials

    def objective(self, trial: optuna.trial.Trial) -> float:
        registry_rot_augmentation = trial.suggest_categorical(
            "registry_rot_augmentation",
            [
                "rot180",
                "rot90",
                "rot270",
                "rot30",
                "rot15",
            ],
        )
        cfg = Config(
            **{
                **asdict(self.cfg),
                **dict(registry_rot_augmentation=[registry_rot_augmentation]),
            }
        )
        score = evaluate(cfg=cfg, fold=self.fold)
        return score

    def __call__(self) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, self.n_trials)
        print(study.best_value, study.best_params)


class Search:
    def __init__(self, cfg: Config, fold: dict, n_trials: int) -> None:
        self.cfg = cfg
        self.fold = fold
        self.n_trials = n_trials

    def objective(self, trial: optuna.trial.Trial) -> float:
        arcface_scale = trial.suggest_float("arcface_scale", 19.0, 25.0)
        arcface_margin = trial.suggest_float("arcface_margin", 5.0, 15.0)
        # embedding_size = trial.suggest_int("embedding_size", 1024, 256 * 7, step=256)
        cfg = Config(
            **{
                **asdict(self.cfg),
                **dict(
                    arcface_scale=arcface_scale,
                    arcface_margin=arcface_margin,
                    # embedding_size=embedding_size,
                ),
            }
        )
        train(cfg=cfg, fold=self.fold)
        score = evaluate(cfg=cfg, fold=self.fold)
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
                    border_mode=cfg.border_mode,
                ),
                ToTensorV2(),
            ],
        )
        self.model = model
        self.transforms = [
            self.transform,
        ]
        for digree in cfg.registry_rot_augmentations:
            self.transforms.append(
                A.Compose(
                    [
                        A.LongestMaxSize(max_size=cfg.image_size),
                        A.PadIfNeeded(
                            min_height=cfg.image_size,
                            min_width=cfg.image_size,
                            border_mode=cfg.border_mode,
                        ),
                        A.Rotate((digree, digree), always_apply=True),
                        ToTensorV2(),
                    ],
                )
            )

        self.all_labels = np.empty(0)
        self.label_map: dict[int, str] = {}

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
        self.label_map = {row["model_no_label"]: row["model_no"] for row in self.rows}
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

    def filter_id(self, embeddings: Tensor) -> list[list[int]]:
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
        res = res[:, : self.cfg.topk]
        return res.tolist()

    def filter_labels_by_image(self, image: Tensor) -> list[str]:
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        with torch.inference_mode():
            embeddings = self.model(image.unsqueeze(0).to(self.model.device))
            ids = self.filter_id(embeddings)[0]
            labels = [self.label_map[i] for i in ids]
        return labels


class Ensemble:
    def __init__(self, topk: int) -> None:
        self.topk = topk
        self.scores = list(range(topk))

    def __call__(self, predictions: list) -> list[str]:
        label_scores: dict[str, float] = dict()
        for preds in predictions:
            for s, p in zip(self.scores, preds):
                if p in label_scores:
                    label_scores[p] += s
                else:
                    label_scores[p] = s
        return [k for k, v in sorted(label_scores.items(), key=lambda x: x[1])][
            : self.topk
        ]


class ScoringService:
    registries: list[Registry]
    ensemble: Ensemble

    @classmethod
    def load_registry(cls, cfg: Config, model_path: str, rows: list[dict]) -> Registry:
        cfg = Config(
            **{
                **asdict(cfg),
                **dict(checkpoint_dir=model_path, pretrained=False, num_workers=0),
            }
        )
        print(cfg.checkpoint_path)
        model = (
            LitModelNoNet.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)
            .eval()
            .cuda()
        )
        registry = Registry(rows=rows, model=model, cfg=cfg)
        registry.create_index()
        return registry

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
        rows = preprocess(image_dir=reference_path, meta_path=reference_meta_path)
        ensemble_config = EnsembleConfig.load("./config/ensemble.yaml")
        cls.registries = [
            cls.load_registry(cfg, model_path=model_path, rows=rows)
            for cfg in ensemble_config.configs
        ]
        cls.ensemble = Ensemble(topk=ensemble_config.topk)
        return True

    @classmethod
    def predict(cls, input: str) -> dict[str, list[str]]:
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        sample_name = os.path.basename(input).split(".")[0]
        # load an image and get the file name
        image = io.imread(input)
        predictions_list = [
            registry.filter_labels_by_image(image) for registry in cls.registries
        ]
        predictions = cls.ensemble(predictions_list)
        output = {sample_name: predictions}
        return output


def setup_fold(cfg: Config) -> dict:
    with open("/app/datasets/train_meta.json", "r") as f:
        meta = json.load(f)
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta=meta,
    )
    folds = kfold(cfg=cfg, rows=rows)
    _, _valid_model_rows = folds[cfg.fold]
    valid_last_2_path_matches = set(
        ["/".join(r["image_path"].split("/")[-2:]) for r in _valid_model_rows]
    )

    with open("/app/datasets/extend_meta.json", "r") as f:
        extend_meta = json.load(f)
    meta = {
        **meta,
        **extend_meta,
    }
    rows = preprocess(
        image_dir="/app/datasets/train",
        meta=meta,
    )
    train_rows = []
    valid_rows = []

    for row in rows:
        last_2_path = "/".join(row["image_path"].split("/")[-2:])
        for p in cfg.hard_samples:
            if row["image_path"].endswith(p):
                train_rows.append(row)
        for valid_last_2_path in valid_last_2_path_matches:
            if last_2_path == valid_last_2_path:
                valid_rows.append(row)
                break
            if not cfg.train_with_flipped and last_2_path.endswith(valid_last_2_path):
                break
        else:
            train_rows.append(row)
    print(len(train_rows))
    print(len(valid_rows))
    return dict(train=train_rows, valid=valid_rows)
