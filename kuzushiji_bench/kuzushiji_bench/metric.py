from vision_tools.interface import TrainBatch
from vision_tools.metric import BoxMAP


class Metric:
    def __init__(self, thresholds: list[float] = [0.5]) -> None:
        self.map = BoxMAP(thresholds)

    @property
    def value(self) -> tuple[float, dict[str, float]]:
        return self.map.value

    def accumulate(self, pred_batch: TrainBatch, gt_batch: TrainBatch) -> None:
        self.map.accumulate(pred_batch["box_batch"], gt_batch["box_batch"])

    def reset(self) -> None:
        self.map.reset()
