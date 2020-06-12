import typing as t
import torch
import json
import numpy as np


from tqdm import tqdm
from pathlib import Path
from logging import getLogger
from torch.utils.data import DataLoader
from app.entities import Annotations
from app.dataset import WheatDataset, plot_row
from app.utils import plot_heatmap

#  from app.dataset import detr_collate_fn as collate_fn
#  from app.models.detr import DETR as NNModel

from app.dataset import collate_fn
from app.models.centernet import CenterNet as NNModel, PreProcess, Criterion

#  from app.models.set_criterion import SetCriterion as Criterion
from app import config

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(
        self, train_data: Annotations, test_data: Annotations, output_dir: Path,
    ) -> None:
        self.model = NNModel().to(device)
        self.criterion = Criterion().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                WheatDataset(test_data),
                shuffle=True,
                batch_size=4,
                drop_last=True,
                collate_fn=collate_fn,
            ),
            "test": DataLoader(
                WheatDataset(dict(list(test_data.items())[:50])),
                batch_size=1,
                collate_fn=collate_fn,
                shuffle=True,
            ),
        }
        self.check_interval = 20
        self.clip_max_norm = config.clip_max_norm
        self.preprocess = PreProcess().to(device)

        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.output_dir.joinpath("checkpoint.json")
        self.best_score = np.inf
        if self.checkpoint_path.exists():
            self.load_checkpoint()

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            (train_loss,) = self.train_one_epoch()
            logger.info(f"{train_loss=}")
            (eval_loss,) = self.eval_one_epoch()
            logger.info(f"{eval_loss=}")
            score = eval_loss
            if score < self.best_score:
                logger.info("update model")
                self.best_score = score
                self.save_checkpoint()

    def train_one_epoch(self) -> t.Tuple[float]:
        self.model.train()
        self.criterion.train()
        epoch_loss = 0
        count = 0
        for samples, targets in tqdm(self.data_loaders["train"]):
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            samples, targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if count % self.check_interval == 0:
                plot_heatmap(
                    targets["heatmap"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("train-tgt-hm.png"),
                )
                plot_heatmap(
                    targets["mask"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("train-tgt-mask.png"),
                )
                plot_heatmap(
                    outputs["heatmap"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("train-src.png"),
                )
        return (epoch_loss / count,)

    @torch.no_grad()
    def eval_one_epoch(self) -> t.Tuple[float]:
        self.model.eval()
        self.criterion.eval()
        epoch_loss = 0
        count = 0
        for samples, targets in tqdm(self.data_loaders["test"]):
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            samples, targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            if count % self.check_interval == 0:
                plot_heatmap(
                    targets["heatmap"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("test-tgt-hm.png"),
                )
                plot_heatmap(
                    targets["mask"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("test-tgt-mask.png"),
                )
                plot_heatmap(
                    outputs["heatmap"][-1][0].detach().cpu(),
                    self.output_dir.joinpath("test-src.png"),
                )

        return (epoch_loss / count,)

    def save_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "w") as f:
            json.dump({"best_score": self.best_score,}, f)
        torch.save(self.model.state_dict(), self.output_dir.joinpath(f"model.pth"))  # type: ignore

    def load_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "r") as f:
            data = json.load(f)
        self.best_score = data["best_score"]
        self.model.load_state_dict(
            torch.load(self.output_dir.joinpath(f"model.pth"))  # type: ignore
        )
