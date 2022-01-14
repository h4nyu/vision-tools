from typing import List, Tuple, Dict
from vision_tools.metric import BoxMAP
from vision_tools.interface import TrainBatch


class Metric:
    def __init__(self, thresholds: List[float] = [0.5]) -> None:
        self.map = BoxMAP(thresholds)

    @property
    def value(self) -> Tuple[float, Dict[str, float]]:
        return self.map.value

    def accumulate(self, pred_batch: TrainBatch, gt_batch: TrainBatch) -> None:
        self.map.accumulate(pred_batch["box_batch"], gt_batch["box_batch"])

    def reset(self) -> None:
        self.map.reset()
