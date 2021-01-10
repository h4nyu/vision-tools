import typing as t
import torch
import numpy as np
from collections import defaultdict
from torch import Tensor
from torchvision.ops.boxes import box_iou
from object_detection.entities import (
    PascalBoxes,
)


def precition(iou_matrix: Tensor, threshold: float) -> float:
    candidates, candidate_ids = (iou_matrix).max(1)
    n_pr, n_gt = iou_matrix.shape
    match_ids = candidate_ids[candidates > threshold]
    (tp,) = torch.unique(match_ids).shape  # type: ignore
    fp = n_pr - tp
    fn = n_gt - tp
    return tp / (fp + tp + fn)


class MeanPrecition:
    def __init__(
        self,
        iou_thresholds: t.List[float] = [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
        ],
    ) -> None:
        self.iou_thresholds = iou_thresholds

    @torch.no_grad()
    def __call__(self, pred_boxes: PascalBoxes, gt_boxes: PascalBoxes) -> float:
        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes) == 0 else 0.0
        if len(pred_boxes) == 0:
            return 0.0

        iou_matrix = box_iou(
            pred_boxes,
            gt_boxes,
        )
        res = np.mean([precition(iou_matrix, t) for t in self.iou_thresholds])
        return res
