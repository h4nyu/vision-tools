class MeanReduceDict:
    def __init__(self, keys: list[str] = []) -> None:
        self.keys = keys
        self.running: dict[str, float] = {}
        self.num_samples = 0

    def accumulate(self, log: dict[str, float]) -> None:
        for k in self.keys:
            self.running[k] = self.running.get(k, 0) + log.get(k, 0)
        self.num_samples += 1

    @property
    def value(self) -> dict[str, float]:
        return {k: self.running.get(k, 0) / max(1, self.num_samples) for k in self.keys}
