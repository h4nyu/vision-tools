import torch
from typing import Set, Any
import numpy as np
from object_detection.entities import PascalBoxes
from torchvision.ops.boxes import box_iou


def auc(
    recall: Any,
    precision: Any,
) -> float:
    rec = np.concatenate(([0.0], recall, [1.0]))
    pre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(pre.size - 1, 0, -1):
        pre[i - 1] = np.maximum(pre[i - 1], pre[i])
    i = np.where(rec[1:] != rec[:-1])[0]
    return np.sum((rec[i + 1] - rec[i]) * pre[i + 1])


class AveragePrecision:
    def __init__(
        self,
        iou_threshold: float,
        eps: float = 1e-8,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.eps = eps

    def __call__(
        self,
        boxes: PascalBoxes,
        gt_boxes: PascalBoxes,
    ) -> float:
        n_gt_box = len(gt_boxes)
        n_box = len(boxes)
        if n_box == 0 or n_gt_box == 0:
            return 0.0

        tp = np.zeros(n_box)
        iou_matrix = box_iou(boxes, gt_boxes)
        confidences, matched_cls_indices = torch.max(iou_matrix, dim=1)
        matched: Set = set()
        for box_id, cls_id in enumerate(matched_cls_indices.to("cpu").numpy()):
            if confidences[box_id] > self.iou_threshold and cls_id not in matched:
                tp[box_id] = 1
                matched.add(cls_id)
        tpc = tp.cumsum()
        fpc = (1 - tp).cumsum()
        recall = fpc / (n_gt_box + self.eps)
        precision = tpc / (tpc + fpc)
        return auc(recall, precision)
