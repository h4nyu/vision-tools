import numpy as np
from torchvision.ops import box_iou
from object_detection.entities.box import PascalBoxes, Labels, Confidences
from .mean_precision import precision
from typing import *


class MeanAveragePrecision:
    def __init__(
        self,
    ) -> None:
        ...

    def __call__(
        self,
        boxes: PascalBoxes,
        labels: Labels,
        gt_boxes: PascalBoxes,
        gt_labels: Labels,
    ) -> float:
        ...


class AveragePrecision:
    def __init__(
        self,
        iou_threshold: float,
        recall_levels:List[float]=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ) -> None:
        self.iou_threshold = iou_threshold
        self.recall_levels = recall_levels

    def __call__(
        self,
        boxes: PascalBoxes,
        gt_boxes: PascalBoxes,
    ) -> float:
        box_len = len(boxes)
        gt_box_len = len(gt_boxes)
        if(box_len == 0 or gt_box_len == 0):
            return 0
        iou_matrix, matched_gt_idx = box_iou(boxes, gt_boxes).max(1)
        matched_gt_idx = matched_gt_idx.to("cpu").numpy()
        box_true = (iou_matrix > self.iou_threshold).to("cpu").numpy()
        matched:Set[int] = set([])
        precisions = np.zeros(box_len)
        recalls = np.zeros(box_len)
        points = np.zeros(len(self.recall_levels))
        for i, (v, gt_idx) in enumerate(zip(box_true, matched_gt_idx)):
            if v and gt_idx not in matched:
                matched.add(gt_idx)
            tp = len(matched)
            precisions[i] = tp / (i + 1)
            recalls[i] = tp / gt_box_len
        for i, v in enumerate(self.recall_levels):
            points[i] = np.max(precisions[recalls > v])
        print(points)

        return 0
