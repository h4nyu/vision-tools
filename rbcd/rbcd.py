from __future__ import annotations

import os
import pathlib
import random
from dataclasses import asdict, dataclass, field
from logging import Logger, getLogger
from multiprocessing import Pool
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
from joblib import Parallel, delayed
from skimage import io
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _WeightedLoss
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


def dicom_to_png(dcm_file: str, image_size: int, image_dir: str = "") -> None:
    patient_id = dcm_file.split("/")[-2]
    image_id = dcm_file.split("/")[-1][:-4]
    dicom = dicomsdl.open(dcm_file)
    img = dicom.pixelData()

    img = (img - img.min()) / (img.max() - img.min())
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(image_dir + "/" + f"{patient_id}_{image_id}.png", img)


def resize_all_images(
    df: pd.Dataframe, input_dir: str, output_dir: str, image_size: int, n_jobs: int = 8
) -> None:
    dcm_file = (
        input_dir
        + "/"
        + df.patient_id.astype(str)
        + "/"
        + df.image_id.astype(str)
        + ".dcm"
    )
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    Parallel(n_jobs=n_jobs)(
        delayed(dicom_to_png)(f, image_size=image_size, image_dir=output_dir)
        for f in tqdm(dcm_file)
    )


class Meter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output: List[Tensor] = []
        self.target: List[Tensor] = []
        self.ids: List[str] = []

    def accumulate(self, output: Tensor, target: Tensor, ids: List[str]) -> None:
        self.output.append(output)
        self.target.append(target)
        self.ids += ids

    def __call__(self) -> Tuple[Tensor, Tensor, List[str]]:
        return (
            torch.cat(self.output),
            torch.cat(self.target),
            self.ids,
        )

    def reset(self) -> None:
        self.output = []
        self.target = []


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prediction_id"] = df["patient_id"].apply(str) + "_" + df["laterality"]
    return df


def create_roi_images(df: pd.DataFrame, output_dir: str) -> None:
    df = df.copy()
    with Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                lambda x: io.imsave(
                    os.path.join(output_dir, x["prediction_id"] + ".png"),
                    extract_roi(x["image"]),
                ),
                df.to_dict("records"),
            ),
            total=len(df),
        ):
            pass


def find_best_threshold(
    target: np.ndarray,
    output: np.ndarray,
    step: float = 0.01,
    start: float = 0.0,
    end: float = 1.0,
    logger: Optional[Logger] = None,
) -> Tuple[float, float]:
    best_thresold = 0.0
    best_score = 0.0
    logger = logger or getLogger(__name__)
    for thresold in np.arange(start, end, step):
        score = pfbeta(target, (output > thresold).astype(float), beta=1)
        logger.info(f"thresold: {thresold:.3f}, score: {score:.3f}")
        if score > best_score:
            best_thresold = thresold
            best_score = score
    return float(best_thresold), float(best_score)


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        smoothing: float = 0.0,
        pos_weight: Any = None,
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets: Tensor, n_labels: int, smoothing: float = 0.0) -> Tensor:
        assert 0.0 <= smoothing < 1.0
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, self.weight, pos_weight=self.pos_weight
        )
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


@dataclass
class InferenceConfig:
    data_path: str = "/store"
    models_dir: str = "models"
    configs_dir: str = "configs"
    batch_size: int = 8
    use_hflip: bool = True
    config_paths: List[str] = field(default_factory=list)

    @property
    def configs(self) -> List[Config]:
        overwrite = dict(
            models_dir=self.models_dir,
            configs_dir=self.configs_dir,
        )
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
    def load(cls, path: str) -> InferenceConfig:
        with open(path) as file:
            obj = yaml.safe_load(file)
        return InferenceConfig(**obj)


@dataclass
class SearchConfig:
    lr_range: Optional[Tuple[float, float]] = None
    ratio_range: Optional[Tuple[float, float]] = None
    weight_decay_range: Optional[Tuple[float, float]] = None
    drop_rate_range: Optional[Tuple[float, float]] = None
    drop_path_rate_range: Optional[Tuple[float, float]] = None
    accumulation_steps_range: Optional[Tuple[int, int]] = None
    use_roi_range: Optional[Tuple[bool, bool]] = None
    affine_scale_range: Optional[Tuple[float, float]] = None
    affine_translate_range: Optional[Tuple[float, float]] = None
    vflip_p_range: Optional[Tuple[float, float]] = None
    affine_p_range: Optional[Tuple[float, float]] = None


@dataclass
class Config:
    name: str
    image_size: int = 256
    in_channels: int = 1
    batch_size: int = 32
    lr: float = 5e-4
    lr_min: float = 1e-6
    warmup_t: int = 2
    optimizer: str = "AdamW"
    total_steps: int = 60_000
    warmup_steps: int = 1_000
    seed: int = 42
    n_splits: int = 5
    model_name: str = "tf_efficientnet_b4_ns"
    pretrained: bool = True
    accumulation_steps: int = 2
    fold: int = 0
    data_path: str = "/app/store"
    models_dir: str = "models"
    configs_dir: str = "configs"
    logs_dir: str = "/app/store/logs"
    scale_limit: float = 0.3
    rotate_limit: float = 15
    validation_steps: int = 1000
    ratio: float = 1.0
    sampling: str = "over"  # deprecated
    device: str = "cuda"
    clip_grad: float = 0.0
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2
    weight_decay: float = 1e-6

    search_limit: Optional[int] = None

    threshold: float = 0.5
    use_roi: bool = False
    use_scheduler: bool = True
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    contrast_p: float = 0.0
    contrast_limit: float = 0.5
    guassian_noise_p: float = 0.25
    elastic_p: float = 0.25
    elastic_alpha: float = 1.0
    elastic_sigma: float = 0.05
    elastic_alpha_affine: float = 0.03
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: int = 0
    affine_p: float = 0.4
    affine_scale: float = 0.2
    affine_translate: float = 0.01
    affine_rotate: float = 30
    affine_shear: float = 20
    cutout_p: float = 0.25
    cutout_holes: int = 5
    cutout_size: float = 0.2
    grid_shuffle_p: float = 0.0

    num_workers: int = 8
    search_config: Optional[SearchConfig] = None
    previous_name: Optional[str] = None
    previous_config: Optional[Config] = None

    @property
    def acc_grad_lr(self) -> float:
        return self.lr / self.accumulation_steps

    @property
    def train_csv(self) -> str:
        return f"{self.data_path}/train.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def valid_csv(self) -> str:
        return f"{self.data_path}/valid.{self.seed}.{self.n_splits}fold{self.fold}.csv"

    @property
    def image_dir(self) -> str:
        return f"{self.data_path}/images_{self.image_size}"

    @property
    def log_dir(self) -> str:
        return f"{self.logs_dir}/{self.name}-{self.seed}-{self.n_splits}fold{self.fold}"

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
        search_config = obj.get("search_config")
        if search_config:
            search_config = SearchConfig(**search_config)
        obj["search_config"] = search_config

        cfg = Config(**obj)
        previous_name = obj.get("previous_name")
        if previous_name:
            cfg.previous_config = Config.load(
                f"{cfg.configs_dir}/{previous_name}.yaml", overwrite
            )
        return cfg


class RdcdPngDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        image_dir: str = "/store/rsna-breast-cancer-256-pngs",
        use_roi: bool = False,
    ) -> None:
        self.df = df
        self.transform = transform
        self.image_dir = image_dir
        self.use_roi = use_roi

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = f"{self.image_dir}/{row['patient_id']}_{row['image_id']}.png"
        image = io.imread(image_path)
        if self.use_roi:
            image = extract_roi(image)
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
            prediction_id=row["prediction_id"],
        )
        return sample


class TrainTransform:
    def __init__(
        self,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        transforms = []
        if cfg.hflip_p > 0:
            transforms.append(A.HorizontalFlip(p=cfg.hflip_p))
        if cfg.vflip_p > 0:
            transforms.append(A.VerticalFlip(p=cfg.vflip_p))
        if cfg.contrast_p > 0:
            transforms.append(
                A.RandomContrast(p=cfg.contrast_p, limit=cfg.contrast_limit)
            )
        if cfg.guassian_noise_p > 0:
            transforms.append(A.GaussNoise(p=cfg.guassian_noise_p))

        transforms.append(
            A.Resize(cfg.image_size, cfg.image_size),
        )
        if cfg.elastic_p > 0:
            transforms.append(
                A.ElasticTransform(
                    p=cfg.elastic_p,
                    alpha=cfg.elastic_alpha * cfg.image_size,
                    sigma=cfg.elastic_sigma * cfg.image_size,
                    alpha_affine=cfg.elastic_alpha_affine * cfg.image_size,
                    border_mode=cfg.border_mode,
                    value=cfg.border_value,
                )
            )
        if cfg.affine_p > 0:
            transforms.append(
                A.Affine(
                    p=cfg.affine_p,
                    scale=(1.0 - cfg.affine_scale, 1.0 + cfg.affine_scale),
                    translate_percent=(-cfg.affine_translate, cfg.affine_translate),
                    rotate=(-cfg.affine_rotate, cfg.affine_rotate),
                    shear=(-cfg.affine_shear, cfg.affine_shear),
                )
            )
        if cfg.grid_shuffle_p > 0:
            transforms.append(A.RandomGridShuffle(p=cfg.grid_shuffle_p))
        if cfg.cutout_p > 0:
            transforms.append(
                A.Cutout(
                    p=cfg.cutout_p,
                    num_holes=cfg.cutout_holes,
                    max_h_size=int(cfg.cutout_size * cfg.image_size),
                    max_w_size=int(cfg.cutout_size * cfg.image_size),
                )
            )

        transforms.extend(
            [
                ToTensorV2(),
            ]
        )

        self.transform = A.Compose(transforms)

    def __call__(self, **kwargs: Any) -> dict:
        return self.transform(**kwargs)


# TrainTransform = lambda cfg: A.Compose(
#     [
#         A.RandomRotate90(),
#         A.ShiftScaleRotate(
#             scale_limit=cfg.scale_limit,
#             rotate_limit=cfg.rotate_limit,
#             p=0.5,
#             border_mode=cfg.border_mode,
#         ),
#     ],
# )

Transform = lambda cfg: A.Compose(
    [
        A.Resize(cfg.image_size, cfg.image_size),
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
        use_roi: bool = False,
    ) -> None:
        self.df = df
        self.transform = transform
        self.image_dir = image_dir
        self.image_size = image_size
        self.use_roi = use_roi

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
        if self.use_roi:
            image = extract_roi(image)
        transformed = self.transform(
            image=image,
        )
        image = transformed["image"].float() / 255.0
        target = (
            torch.tensor([row["cancer"]]) if row.get(["cancer"]) is not None else None
        )
        if target is None:
            return dict(
                image=image,
                patient_id=row["patient_id"],
                image_id=row["image_id"],
            )
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
    def __init__(
        self,
        name: str,
        in_channels: int,
        pretrained: bool = True,
        drop_rate: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=1,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
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
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
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

    def __call__(self, limit: Optional[int] = None) -> float:
        cfg = self.cfg
        if cfg.previous_config is not None:
            self.net.load_state_dict(torch.load(cfg.previous_config.model_path))
        train_df = read_csv(cfg.train_csv)
        if limit is not None:
            train_df = train_df[:limit]
        valid_df = read_csv(cfg.valid_csv)
        if limit is not None:
            valid_df = valid_df[:limit]
        train_dataset = RdcdPngDataset(
            df=train_df,
            transform=TrainTransform(cfg),
            image_dir=cfg.image_dir,
            use_roi=cfg.use_roi,
        )
        sampler = OverSampler(train_dataset, shuffle=True, ratio=cfg.ratio)
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.acc_grad_lr, weight_decay=cfg.weight_decay
        )
        if cfg.use_scheduler:
            scheduler = timm.scheduler.CosineLRScheduler(
                optimizer,
                t_initial=cfg.total_steps,
                warmup_t=cfg.warmup_steps,
                lr_min=cfg.lr_min,
                t_in_epochs=True,
            )
        else:
            scheduler = None
        criterion = nn.BCEWithLogitsLoss()
        best_score = 0.0
        best_loss = np.inf
        validate = Validate(self.cfg)
        epochs = cfg.total_steps // len(train_loader)
        for epoch in range(epochs):
            meter = Meter()
            train_loss = 0.0
            train_score = 0.0
            for batch in tqdm(train_loader):
                if self.iteration % cfg.validation_steps == 0:
                    val_res = validate(self.model)
                    self.writer.add_scalar(
                        "valid/loss", val_res["loss"], self.iteration
                    )
                    self.writer.add_scalar(
                        "valid/score", val_res["score"], self.iteration
                    )
                    self.writer.add_scalar(
                        "valid/binarized_score",
                        val_res["binarized_score"],
                        self.iteration,
                    )
                    self.writer.add_scalar(
                        "valid/threshold", val_res["threshold"], self.iteration
                    )
                    self.writer.add_scalar("valid/auc", val_res["auc"], self.iteration)
                    self.writer.add_scalar(
                        "valid/agg_score", val_res["score"], self.iteration
                    )
                    self.writer.add_scalar(
                        "valid/agg_binarized_score",
                        val_res["agg_binarized_score"],
                        self.iteration,
                    )
                    self.writer.add_scalar(
                        "valid/agg_threshold", val_res["agg_threshold"], self.iteration
                    )
                    self.writer.add_scalar(
                        "valid/agg_auc", val_res["auc"], self.iteration
                    )
                    self.writer.add_scalar(
                        "lr", optimizer.param_groups[0]["lr"], self.iteration
                    )
                    self.writer.flush()
                    if val_res["loss"] < best_loss:
                        best_loss = val_res["loss"]
                        torch.save(
                            self.model.state_dict(),
                            self.cfg.model_path,
                        )
                        self.logger.info(f"save model: {self.cfg.model_path}")
                    self.logger.info(
                        f"iteration: {self.iteration} valid/loss: {val_res['loss']}, valid/agg_binarized_score: {val_res['agg_binarized_score']} valid/agg_score: {val_res['agg_score']}"
                    )

                self.model.train()
                self.iteration += 1
                image = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                optimizer.zero_grad()
                with autocast():
                    output = self.model(image)
                    loss = (
                        criterion(output, target.float()) / self.cfg.accumulation_steps
                    )
                self.scaler.scale(loss).backward()
                if self.iteration % self.cfg.accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.clip_grad
                        )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                if scheduler is not None:
                    scheduler.step(self.iteration)
                train_loss += loss.item()
                meter.accumulate(output.sigmoid(), target, batch["prediction_id"])

            train_loss /= len(train_loader)
            output, target, prediction_id = meter()
            df = pd.DataFrame(
                {
                    "prediction_id": prediction_id,
                    "target": target.flatten().cpu().detach().numpy(),
                    "pred": output.flatten().cpu().detach().numpy(),
                }
            )
            train_agg_score = pfbeta(
                df.target.values,
                df.pred.values,
            )
            df = (
                df.groupby("prediction_id")
                .agg({"target": "max", "pred": "max"})
                .reset_index()
            )
            train_agg_score = pfbeta(
                df.target.values,
                df.pred.values,
            )
            self.writer.add_scalar("train/loss", train_loss, self.iteration)
            self.writer.add_scalar("train/agg_score", train_agg_score, self.iteration)
            self.writer.add_scalar("train/score", train_score, self.iteration)
        return best_score


class Validate:
    def __init__(
        self,
        cfg: Config,
        df: Optional[pd.DataFrame] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        valid_df = df if df is not None else read_csv(cfg.valid_csv)
        valid_dataset = RdcdPngDataset(
            df=valid_df,
            transform=Transform(cfg),
            image_dir=cfg.image_dir,
            use_roi=cfg.use_roi,
        )
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size * 2,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.logger = logger or getLogger(cfg.name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(
        self, net: nn.Module, enable_find_threshold: bool = False
    ) -> Dict[str, float]:
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
                meter.accumulate(
                    output.sigmoid().flatten(), target.flatten(), batch["prediction_id"]
                )
        epoch_loss /= len(self.valid_loader)
        output, target, prediction_id = meter()
        df = pd.DataFrame(
            {
                "prediction_id": prediction_id,
                "target": target.flatten().cpu().numpy(),
                "pred": output.flatten().cpu().numpy(),
            }
        )
        score = pfbeta(
            df.target.values,
            df.pred.values,
        )
        auc = roc_auc_score(df.target.values, df.pred.values)
        threshold, binarized_score = find_best_threshold(
            df.target.values, df.pred.values
        )
        df = (
            df.groupby("prediction_id")
            .agg({"target": "max", "pred": "max"})
            .reset_index()
        )
        agg_score = pfbeta(
            df.target.values,
            df.pred.values,
        )
        agg_auc = roc_auc_score(df.target.values, df.pred.values)
        agg_threshold, agg_binarized_score = find_best_threshold(
            df.target.values, df.pred.values
        )
        return dict(
            loss=epoch_loss,
            score=score,
            binarized_score=binarized_score,
            threshold=threshold,
            auc=auc,
            agg_score=agg_score,
            agg_binarized_score=agg_binarized_score,
            agg_threshold=agg_threshold,
            agg_auc=agg_auc,
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
        search_config = self.cfg.search_config
        overwrite_config: Dict[str, Any] = {}
        if search_config is None:
            raise ValueError("search_config is None")
        if search_config.lr_range is not None:
            lr = trial.suggest_float("lr", *search_config.lr_range)
            overwrite_config["lf"] = lr
        if search_config.ratio_range is not None:
            ratio = trial.suggest_float("ratio", *search_config.ratio_range)
            overwrite_config["ratio"] = ratio
        if search_config.drop_rate_range is not None:
            drop_rate = trial.suggest_float("drop_rate", *search_config.drop_rate_range)
            overwrite_config["drop_rate"] = drop_rate
        if search_config.drop_path_rate_range is not None:
            drop_path_rate = trial.suggest_float(
                "drop_path_rate", *search_config.drop_path_rate_range
            )
            overwrite_config["drop_path_rate"] = drop_path_rate
        if search_config.weight_decay_range is not None:
            weight_decay = trial.suggest_float(
                "weight_decay", *search_config.weight_decay_range
            )
            overwrite_config["weight_decay"] = weight_decay
        if search_config.accumulation_steps_range is not None:
            accumulation_steps = trial.suggest_int(
                "accumulation_steps", *search_config.accumulation_steps_range
            )
            overwrite_config["accumulation_steps"] = accumulation_steps
        if search_config.use_roi_range is not None:
            use_roi = trial.suggest_categorical("use_roi", search_config.use_roi_range)
            overwrite_config["use_roi"] = use_roi
        if search_config.affine_scale_range is not None:
            affine_scale = trial.suggest_float(
                "affine_scale", *search_config.affine_scale_range
            )
            overwrite_config["affine_scale"] = affine_scale
        if search_config.affine_translate_range is not None:
            affine_translate = trial.suggest_float(
                "affine_translate", *search_config.affine_translate_range
            )
            overwrite_config["affine_translate"] = affine_translate
        if search_config.vflip_p_range is not None:
            vflip_p = trial.suggest_float("vflip_p", *search_config.vflip_p_range)
            overwrite_config["vflip_p"] = vflip_p
        if search_config.affine_p_range is not None:
            affine_p = trial.suggest_float("affine_p", *search_config.affine_p_range)
            overwrite_config["affine_p"] = affine_p
        name = f"{self.cfg.name}_{trial.number}"
        overwrite_config["name"] = name
        cfg = Config(
            **{
                **asdict(self.cfg),
                **overwrite_config,
            }
        )
        self.logger.info(f"trial: {trial.number}, cfg: {cfg}")
        train = Train(cfg, logger=self.logger)
        validate = Validate(cfg, logger=self.logger)
        train(self.limit)
        model = Model.load(cfg).to(cfg.device)
        res = validate(model)
        self.logger.info(res)
        return res["agg_binarized_score"]

    def __call__(self) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, self.n_trials)
        self.logger.info(
            f"best_params: {study.best_params} best_value: {study.best_value}"
        )


class EnsembleInference:
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

    def vote(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df["cancer"] = df[cols].mean(axis=1)
        df = (
            df[["prediction_id", "cancer"]]
            .groupby("prediction_id")
            .agg({"cancer": "max"})
            .reset_index()
        )
        return df

    def __call__(self) -> pd.DataFrame:
        df = read_csv(self.cfg.test_csv)
        output_cols = []
        for cfg in self.cfg.configs:
            self.logger.info(f"cfg: {cfg}")
            inference = Inference(cfg, inference_cfg=self.cfg, logger=self.logger)
            output = inference(df)
            output_cols.append(cfg.name)
            df[cfg.name] = output > cfg.threshold
        submission_df = self.vote(df[["prediction_id"] + output_cols], output_cols)
        return submission_df


def extract_roi(x: np.ndarray) -> np.ndarray:
    x = x[5:-5, 5:-5]
    output = cv2.connectedComponentsWithStats(
        (x > 20).astype(np.uint8)[:, :], 8, cv2.CV_32S
    )
    stats = output[2]
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    roi = x[y1 : y1 + h, x1 : x1 + w]
    return roi


class Inference:
    def __init__(
        self,
        cfg: Config,
        inference_cfg: InferenceConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.inference_cfg = inference_cfg
        self.logger = logger or getLogger(__name__)

    @property
    def num_workers(self) -> int:
        return os.cpu_count() or 0

    @torch.no_grad()
    def __call__(self, df: pd.Dataframe) -> np.ndarray:
        dataset = RdcdDicomDataset(
            df=df,
            transform=Transform(self.cfg),
            image_dir=self.inference_cfg.image_dir,
            use_roi=self.cfg.use_roi,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.inference_cfg.batch_size,
            num_workers=self.num_workers,
        )
        model = Model.load(self.cfg).to(self.cfg.device)
        outputs = []
        for batch in tqdm(loader):
            image = batch["image"].to(self.cfg.device)
            output = model(image)
            outputs.append(output.sigmoid().detach().cpu().numpy())
        return np.concatenate(outputs)
