import torch
from typing import Set, Any, Tuple, List
import numpy as np
from object_detection.entities import PascalBoxes, Confidences
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
        self.tp_list: List[Any] = []
        self.confidence_list: List[Any] = []
        self.n_gt_box = 0

    def reset(self) -> None:
        self.n_gt_box = 0
        self.confidence_list = []
        self.tp_list = []

    def add(
        self,
        boxes: PascalBoxes,
        confidences: Confidences,
        gt_boxes: PascalBoxes,
    ) -> None:
        n_gt_box = len(gt_boxes)
        n_box = len(boxes)
        if n_box == 0:
            return
        tp = np.zeros(n_box)
        iou_matrix = box_iou(boxes, gt_boxes)
        ious, matched_cls_indices = torch.max(iou_matrix, dim=1)
        matched: Set = set()
        for box_id, cls_id in enumerate(matched_cls_indices.to("cpu").numpy()):
            if ious[box_id] > self.iou_threshold and cls_id not in matched:
                tp[box_id] = 1
                matched.add(cls_id)
        self.tp_list.append(tp)
        self.confidence_list.append(confidences.to("cpu").numpy())
        self.n_gt_box += n_gt_box

    def __call__(self) -> float:
        if len(self.confidence_list) == 0:
            return 0.0
        sort_indices = np.argsort(-np.concatenate(self.confidence_list))
        tp = np.concatenate(self.tp_list)[sort_indices]
        tpc = tp.cumsum()
        fpc = (1 - tp).cumsum()
        recall = tpc / (self.n_gt_box + self.eps)
        precision = tpc / (tpc + fpc)
        return auc(recall, precision)
