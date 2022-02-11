

class MeanAP:
    def __init__(
        self
        n: int = 5,
        cutoff: int = 5,
    ) -> None:
        self.n = n
        self.cutoff = cutoff
        self.reset()

    def reset(self) -> None:
        self.n_sample = 0

    def update(self, ap):
        self.mAP += ap
        self.num_inst += 1

    def get(self):
        return self.mAP / self.num_inst
