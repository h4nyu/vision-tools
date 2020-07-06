import typing as t
import torch
import numpy as np
from collections import defaultdict
from torch import Tensor
from torchvision.ops.boxes import box_iou
from object_detection.entities import yolo_to_pascal, YoloBoxes


def precition(iou_matrix: Tensor, threshold: float) -> float:
    candidates, candidate_ids = (iou_matrix).max(1)
    n_pr, n_gt = iou_matrix.shape
    match_ids = candidate_ids[candidates > threshold]
    fp = n_pr - len(match_ids)
    (tp,) = torch.unique(match_ids).shape  # type: ignore
    fn = n_gt - tp
    return tp / (fp + tp + fn)


class MeanPrecition:
    def __init__(
        self, iou_thresholds: t.List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
    ) -> None:
        self.iou_thresholds = iou_thresholds

    def __call__(self, pred_boxes: YoloBoxes, gt_boxes: YoloBoxes) -> float:
        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes) == 0 else 0.0
        if len(pred_boxes) == 0:
            return 0.0

        iou_matrix = box_iou(
            yolo_to_pascal(pred_boxes, (1, 1)), yolo_to_pascal(gt_boxes, (1, 1)),
        )
        res = np.mean([precition(iou_matrix, t) for t in self.iou_thresholds])
        return res