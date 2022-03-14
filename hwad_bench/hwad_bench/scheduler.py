from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.optim.lr_scheduler import (  # type: ignore
    OneCycleLR,
    ReduceLROnPlateau,
    _LRScheduler,
)


class WarmupReduceLROnPlaetou(_LRScheduler):
    def __init__(
        self,
        optimizer: Any,
        warmup_steps: int,
        max_lr: float,
        min_lr: float = 1e-8,
        factor: float = 0.1,
        patience: int = 10,
        cooldown: int = 0,
        mode: str = "min",
    ) -> None:
        self.warmup = OneCycleLR(
            optimizer,
            total_steps=warmup_steps,
            max_lr=max_lr,
            pct_start=1.0,
        )
        self.warmup_steps = warmup_steps
        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            cooldown=cooldown,
            min_lr=min_lr,
        )
        self.last_epoch = -1

    def step(self, metrics: Union[Tensor, float]) -> None:  # type: ignore
        self.last_epoch += 1

        if self.last_epoch < self.warmup_steps:
            self.warmup.step(self.last_epoch)
        else:
            self.plateau.step(metrics, epoch=self.last_epoch - self.warmup_steps)

    def get_last_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            return self.warmup.get_last_lr()
        else:
            return [x["lr"] for x in self.plateau.optimizer.param_groups]  # type: ignore
