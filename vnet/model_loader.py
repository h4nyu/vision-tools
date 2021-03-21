import torch
from logging import getLogger
from typing import *
from typing_extensions import Literal
import operator
import math
from torch import nn
from pathlib import Path
import json

logger = getLogger(__name__)


WatchMode = Literal["min", "max"]


class BestWatcher:
    def __init__(
        self,
        mode: WatchMode = "min",
        min_delta: float = 0.0,
        ema: bool = False,
        alpha: float = 1.0,
    ) -> None:
        self.mode = mode
        self.min_delta = min_delta
        self.ema = ema
        self.alpha = alpha
        if mode == "min":
            self.op = operator.lt
            self.best = math.inf
        else:
            self.op = operator.gt
            self.best = -math.inf
        self.prev_metrics = math.nan

    def step(self, metrics: float) -> bool:
        if self.ema and not math.isnan(self.prev_metrics):
            metrics = self.prev_metrics * self.alpha + metrics * (1 - self.alpha)
        self.prev_metrics = metrics
        if self.op(metrics - self.min_delta, self.best):
            self.best = metrics
            return True
        else:
            return False


class ModelLoader:
    def __init__(
        self,
        out_dir: str,
        best_watcher: BestWatcher,
        key: str = "checkpoint",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.key = key
        self.checkpoint_file = self.out_dir / f"{self.key}.json"
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.model_path = self.out_dir / f"{self.key}.pth"
        self.best_watcher = best_watcher

    def check_point_exists(self) -> bool:
        return self.checkpoint_file.exists() and self.model_path.exists()

    def load_if_needed(self, model: nn.Module) -> nn.Module:
        if self.check_point_exists():
            model = self._load(model)
        return model

    def _load(self, model: nn.Module) -> nn.Module:
        logger.info(f"load model from {self.model_path}")
        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)
        model.load_state_dict(torch.load(self.model_path))  # type: ignore
        self.best_watcher.step(data)
        return model

    def save_if_needed(self, model: nn.Module, metric: float) -> None:
        if self.best_watcher.step(metric):
            self._save(model, metric)

    def _save(self, model: nn.Module, metric: float) -> None:
        with open(self.checkpoint_file, "w") as f:
            json.dump(metric, f)
        torch.save(model.state_dict(), self.model_path)  # type: ignore
        logger.info(f"save model to {self.model_path}")
