from typing import Any


class MeanReduceDict:
    def __init__(self) -> None:
        self.running: dict[str, float] = {}
        self.num_samples = 0

    def accumulate(self, log: Any) -> None:
        for k in log.keys():
            self.running[k] = self.running.get(k, 0) + log.get(k, 0)
        self.num_samples += 1

    def reset(self) -> None:
        self.running = {}
        self.num_samples = 0

    @property
    def value(self) -> dict[str, float]:
        return {
            k: self.running.get(k, 0) / max(1, self.num_samples)
            for k in self.running.keys()
        }
