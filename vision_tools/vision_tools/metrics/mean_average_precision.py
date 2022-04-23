from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from vision_tools.metrics.average_precision import AveragePrecision


class MeanAveragePrecision:
    def __init__(
        self, num_classes: int, iou_threshold: float, eps: float = 1e-8
    ) -> None:
        self.ap = AveragePrecision(iou_threshold, eps)
        self.aps = {k: AveragePrecision(iou_threshold, eps) for k in range(num_classes)}
        self.eps = eps

    def reset(self) -> None:
        for v in self.aps.values():
            v.reset()

    @torch.no_grad()
    def add(
        self,
        boxes: Tensor,
        confidences: Tensor,
        labels: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> None:
        unique_gt_labels = set(np.unique(gt_labels.to("cpu").numpy()))
        unique_labels = set(np.unique(labels.to("cpu").numpy()))
        for k in unique_labels | unique_gt_labels:
            ap = self.aps[k]
            ap.add(
                boxes=boxes[labels == k],
                confidences=confidences[labels == k],
                gt_boxes=gt_boxes[gt_labels == k],
            )

    @torch.no_grad()
    def __call__(self) -> Tuple[float, Dict[int, float]]:
        aps = {k: v() for k, v in self.aps.items()}
        return np.fromiter(aps.values(), dtype=float).mean(), aps
