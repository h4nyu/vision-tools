import typing as t
import random
import numpy as np
import torch
from app.models.centernet import (
    CenterNet as NNModel,
    Criterion,
    PreProcess,
    PostProcess,
    Visualize,
)
from app import config
from app.utils import ModelLoader
from torch import nn
from torch.utils.data import DataLoader
from app.meters import BestWatcher
from logging import getLogger

logger = getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        best_watcher: BestWatcher,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.model_loader = model_loader
        self.model = model_loader.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = Criterion()
        self.preprocess = PreProcess(self.device)
        self.post_process = PostProcess()
        self.best_watcher = best_watcher
        self.train_visalize = Visualize("train")
        self.test_visalize = Visualize("test")

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()
            self.eval_one_epoch()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for samples, targets, ids in loader:
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.criterion(outputs, cri_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.eval()
        loader = self.test_loader
        visualize = self.test_visalize
        for samples, targets, ids in loader:
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss = self.criterion(outputs, cri_targets)
            if self.best_watcher.step(loss.item()):
                self.model_loader.save()
            preds = self.post_process(outputs, ids, samples)
            visualize(outputs, preds, samples)
