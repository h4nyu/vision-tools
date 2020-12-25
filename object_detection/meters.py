import numpy as np
import operator
import math
from logging import getLogger
from typing_extensions import Literal


class EMAMeter:
    def __init__(self, alpha: float = 0.99) -> None:
        self.alpha = alpha
        self.ema = np.nan
        self.count = 0

    def update(self, value: float) -> None:
        if np.isnan(self.ema):
            self.ema = value
        else:
            self.ema = self.ema * self.alpha + value * (
                1.0 - self.alpha
            )
        self.count += 1

    def get_value(self) -> float:
        return self.ema

    def reset(self) -> None:
        self.ema = np.nan


class MeanMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.sum += value * count
        self.count += count

    def get_value(self) -> float:
        return self.sum / max(self.count, 1)

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0
