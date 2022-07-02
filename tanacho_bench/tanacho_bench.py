from __future__ import annotations

import json
import os
from typing import Any, Callable

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchmetrics
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from vision_tools.utils import Checkpoint, ToDevice, seed_everything


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
        return self.score_sum / min(self.update_count, 1)

    def reset(
        self,
    ) -> None:
        self.update_count = 0
        self.score_sum = 0


def preprocess(path: str, image_dir: str) -> dict:
    rows = []
    with open(path, "r") as f:
        meta = json.load(f)
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
        A.LongestMaxSize(max_size=cfg["image_size"]),
        A.PadIfNeeded(
            min_height=cfg["image_size"],
            min_width=cfg["image_size"],
            border_mode=cv2.BORDER_REPLICATE,
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
        A.LongestMaxSize(max_size=cfg["image_size"]),
        A.PadIfNeeded(
            min_height=cfg["image_size"],
            min_width=cfg["image_size"],
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
        image = cv2.imread(row["image_path"])
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


def preview_dataset(cfg: dict, rows: list[dict], path: str) -> None:
    dataset = TanachoDataset(
        rows=rows,
        transform=TrainTransform(cfg),
    )
    grid = make_grid([dataset[i]["sample"]["image"] for i in range(10)])
    save_image(grid, path)


def kfold(cfg: dict, rows: list[dict]) -> list[dict]:
    """fold by model_no"""
    skf = StratifiedKFold(n_splits=cfg["kfold"])
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
        return embeddings


class LitModelNoNet(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = Net(
            name=cfg["model_name"],
            embedding_size=cfg["embedding_size"],
        )
        self.arcface = ArcFaceLoss(
            num_classes=cfg["num_model_nos"],
            embedding_size=cfg["embedding_size"],
        )
        self.save_hyperparameters()

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        sample, batch_idx = args
        batch = sample["sample"]
        embeddings = self.net(batch["image"])
        loss = self.arcface(embeddings, batch["model_no_label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # def predict_step(self, *args: Any, **kwargs: Any) -> Any:
    #     sample, batch_idx = args
    #     batch = sample["sample"]
    #     out = self.net(batch["image"])
    #     return out.softmax(dim=-1).argmax(dim=1), sample["row"]

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        sample, batch_idx = args
        batch = sample["sample"]
        embeddings = self.net(batch["image"])
        loss = self.arcface(embeddings, batch["model_no_label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        return optim.AdamW(
            self.net.parameters(),
            lr=self.cfg["head_lr"],
        )


def train(cfg: dict, fold: dict) -> None:
    seed_everything(cfg["seed"])
    train_dataset = TanachoDataset(
        rows=fold["train"],
        transform=TrainTransform(cfg),
    )
    valid_dataset = TanachoDataset(
        rows=fold["valid"],
        transform=TrainTransform(cfg),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor=cfg["monitor"],
                mode=cfg["mode"],
                dirpath="checkpoints",
                filename=cfg["color_model_filename"],
            ),
            pl.callbacks.EarlyStopping(
                monitor=cfg["monitor"], mode=cfg["mode"], patience=cfg["patience"]
            ),
        ],
    )
    trainer.fit(
        LitModelNoNet(cfg),
        train_loader,
        valid_loader,
    )


def evaluate(cfg: dict, rows: list[dict], encoders: dict) -> None:
    dataset = TanachoDataset(
        rows=rows,
        transform=InferenceTransform(cfg),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    ap = torchmetrics.LabelRankingAveragePrecision()
    # color_model = LitColorNet.load_from_checkpoint(
    #     f"checkpoints/{cfg['color_model_filename']}.ckpt", cfg=cfg
    # )
    # category_model = LitCategoryNet.load_from_checkpoint(
    #     f"checkpoints/{cfg['category_model_filename']}.ckpt", cfg=cfg
    # )
    # trainer = pl.Trainer( precision=16,
    #     accelerator="gpu",
    # )
    # color_predictions = trainer.predict(color_model, dataloader)
    # category_predictions = trainer.predict(category_model, dataloader)
    # pred_colors = []
    # pred_categories = []
    # gt_colors = []
    # gt_categories = []
    # pred_combined = []
    # gt_combined = []
    # for (color_preds, batch_rows), (category_preds, _) in zip(
    #     color_predictions, category_predictions
    # ):
    #     pred_colors += color_preds.tolist()
    #     pred_categories += category_preds.tolist()
    #     gt_colors += batch_rows["color_label"].tolist()
    #     gt_categories += batch_rows["category_label"].tolist()
    #     gt_combined += batch_rows["combined_label"].tolist()
    # color_acc = accuracy(torch.tensor(pred_colors), torch.tensor(gt_colors))
    # category_acc = accuracy(torch.tensor(pred_categories), torch.tensor(gt_categories))
    # for color, category in zip(
    #     encoders["color"].inverse_transform(pred_colors),
    #     encoders["category"].inverse_transform(pred_categories),
    # ):
    #     pred_combined.append(f"{category}_{color}")
    # combined_acc = accuracy(
    #     torch.tensor(encoders["combined"].transform(pred_combined)),
    #     torch.tensor(gt_combined),
    # )
    # print(f"Color accuracy: {color_acc}")
    # print(f"Category accuracy: {category_acc}")
    # print(f"Combined accuracy: {combined_acc}")
