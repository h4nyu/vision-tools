import numpy as np
from logging import getLogger


class EMAMeter:
    def __init__(
        self, name: str = "", alpha: float = 0.99, log_period: int = 10
    ) -> None:
        self.alpha = alpha
        self.ema = np.nan
        self.logger = getLogger(name)
        self.count = 0
        self.log_period = log_period

    def update(self, value: float) -> None:
        if np.isnan(self.ema):
            self.ema = value
        else:
            self.ema = self.ema * self.alpha + value * (1.0 - self.alpha)
        self.count += 1
        if self.count % self.log_period == 0:
            self.count = 0
            self.logger.info(f"{self.ema:.4f}")

    def get_value(self) -> float:
        return self.ema

    def reset(self) -> None:
        self.ema = np.nan
        self.logger.info("reset")
