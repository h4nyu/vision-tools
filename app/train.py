import typing as t
import torch
import json
import numpy as np

from pathlib import Path
from logging import getLogger
from torch.utils.data import DataLoader
from app.entities import Annotations
from app.dataset import WheatDataset
from app.utils import plot_heatmap

#  from app.dataset import detr_collate_fn as collate_fn
#  from app.models.detr import DETR as NNModel

from app.dataset import collate_fn
from app.models.centernet import (
    CenterNet as NNModel,
    PreProcess,
    Criterion,
    VisualizeHeatmap,
    PostProcess,
    Evaluate,
)
from app import config

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


def load_checkpoint(output_dir: Path, model: NNModel) -> t.Tuple[NNModel, float]:
    with open(output_dir.joinpath("checkpoint.json"), "r") as f:
        data = json.load(f)
    best_score = data["best_score"]
    model.load_state_dict(
        torch.load(output_dir.joinpath(f"model.pth"))  # type: ignore
    )
    return model, best_score


class Trainer:
    def __init__(
        self, train_data: Annotations, test_data: Annotations, output_dir: Path,
    ) -> None:
        self.model = NNModel().to(device)
        self.train_cri = Criterion("train").to(device).train()
        self.test_cri = Criterion("test").to(device).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.evaluate = Evaluate()
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                WheatDataset(train_data),
                shuffle=True,
                batch_size=config.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers,
            ),
            "test": DataLoader(
                WheatDataset(test_data, "test"),
                batch_size=config.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                num_workers=config.num_workers,
            ),
        }
        self.visualizes = {
            "test": VisualizeHeatmap(output_dir, "test"),
            "train": VisualizeHeatmap(output_dir, "train"),
        }
        self.check_interval = 20
        self.preprocess = PreProcess().to(device)

        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.output_dir.joinpath("checkpoint.json")
        self.best_score = 0.0
        if self.checkpoint_path.exists():
            self.model, self.best_score = load_checkpoint(self.output_dir, self.model)

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            (train_loss,) = self.train_one_epoch()
            logger.info(f"{train_loss=}")
            (eval_loss, score) = self.eval_one_epoch()
            logger.info(f"{eval_loss=},{score=}")
            if score > self.best_score:
                logger.info("update model")
                self.best_score = score
                self.save_checkpoint()

    def train_one_epoch(self) -> t.Tuple[float]:
        self.model.train()
        epoch_loss = 0
        count = 0
        loader = self.data_loaders["train"]
        for samples, targets in loader:
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.train_cri(outputs, cri_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        self.visualizes["train"](samples, outputs, targets)
        return (epoch_loss / count,)

    @torch.no_grad()
    def eval_one_epoch(self) -> t.Tuple[float, float]:
        self.model.eval()
        epoch_loss = 0
        count = 0
        loader = self.data_loaders["test"]
        score = 0.0
        for samples, targets in loader:
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.test_cri(outputs, cri_targets)
            epoch_loss += loss.item()
            score += self.evaluate(outputs, targets)
        self.visualizes["test"](samples, outputs, targets)
        return (epoch_loss / count, score / count)

    def save_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "w") as f:
            json.dump({"best_score": self.best_score,}, f)
        torch.save(self.model.state_dict(), self.output_dir.joinpath(f"model.pth"))  # type: ignore


class Preditor:
    def __init__(self, output_dir: Path, data: Annotations,) -> None:
        self.model, _ = load_checkpoint(output_dir, NNModel().to(device),)
        self.data_loader = DataLoader(
            WheatDataset(data, "test"),
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=config.num_workers,
        )
        self.visualize = VisualizeHeatmap(output_dir, "pred")
        self.preprocess = PreProcess().to(device)
        self.postprocess = PostProcess()

    @torch.no_grad()
    def __call__(self) -> Annotations:
        annotations: Annotations = dict()
        for samples, targets, ids in self.data_loader:
            samples = samples.to(device)
            samples, _ = self.preprocess((samples, targets))
            outputs = self.model(samples)
            batch_annots = self.postprocess(outputs, ids)
            self.visualize(samples, outputs, targets)
            annotations.update(batch_annots)
        return annotations
