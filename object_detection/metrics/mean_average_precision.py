import numpy as np
from typing import Tuple, List, Dict
from object_detection.metrics.average_precision import AveragePrecision
from object_detection.entities import PascalBoxes, Labels


class MeanAveragePrecision:
    def __init__(
        self, classes: List[int], iou_threshold: float, eps: float = 1e-8
    ) -> None:
        self.classes = classes
        self.ap = AveragePrecision(iou_threshold, eps)
        self.eps = eps

    def __call__(
        self,
        labels: Labels,
        boxes: PascalBoxes,
        gt_labels: Labels,
        gt_boxes: PascalBoxes,
    ) -> Tuple[float, Dict[int, float]]:
        aps = {
            k: self.ap(
                boxes=PascalBoxes(boxes[labels == k]),
                gt_boxes=PascalBoxes(gt_boxes[gt_labels == k]),
            )
            for k in self.classes
        }
        return np.mean(aps.values()), aps
