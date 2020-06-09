import typing as t
import torch
import json
import numpy as np


from tqdm import tqdm
from pathlib import Path
from logging import getLogger
from torch.utils.data import DataLoader
from app.entities import Annotations
from app.dataset import WheatDataset, collate_fn, plot_row
from app.models.detr import DETR as NNModel
from app.models.set_criterion import SetCriterion as Criterion
from app import config

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(
        self, train_data: Annotations, test_data: Annotations, output_dir: Path,
    ) -> None:
        self.model = NNModel().to(device)
        self.criterion = Criterion(num_classes=config.num_classes).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),)
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                WheatDataset(train_data),
                shuffle=True,
                batch_size=8,
                drop_last=True,
                collate_fn=collate_fn,
            ),
            "test": DataLoader(
                WheatDataset(test_data), batch_size=8, collate_fn=collate_fn,
                shuffle=True,
            ),
        }

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
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return (epoch_loss / count,)

    def eval_one_epoch(self) -> t.Tuple[float]:
        self.model.eval()
        self.criterion.eval()
        epoch_loss = 0
        count = 0
        with torch.no_grad():
            for samples, targets in tqdm(self.data_loaders["test"]):
                count += 1
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)
                epoch_loss += loss.item()
            plot_row(
                samples.decompose()[0][-1].cpu(),
                outputs["pred_boxes"][-1].cpu(),
                self.output_dir.joinpath("eval.png"),
                targets[-1]["boxes"].cpu(),
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
