import torch
from torch import Tensor
from typing import List, Tuple, Dict
from vision_tools.metric import BoxMAP
from torchvision.ops import box_iou
from vision_tools.interface import TrainBatch
import numpy as np


class BoxF2:
    def __init__(
        self, iou_thresholds: List[float] = list(np.arange(0.3, 0.85, 0.05))
    ) -> None:
        self.beta = 2.0
        self.iou_thresholds = iou_thresholds
        self.correct = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    def precision(self, tp: int, fp: int, fn: int) -> float:
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, tp: int, fp: int, fn: int) -> float:
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f_beat(self, precision: float, recall: float, beta: float) -> float:
        if precision + recall == 0:
            return 0.0
        return (
            (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
        )

    @property
    def value(self) -> Tuple[float, Dict[str, float]]:
        precision = self.precision(**self.correct)
        recall = self.recall(**self.correct)
        f2 = self.f_beat(precision, recall, self.beta)
        return f2, dict(
            f2=f2,
            precision=precision,
            recall=recall,
        )

    def accumulate(self, pred_batch: TrainBatch, gt_batch: TrainBatch) -> None:
        for pred_boxes, gt_boxes in zip(pred_batch["box_batch"], gt_batch["box_batch"]):
            for iou_threshold in self.iou_thresholds:
                correct = self.correct_at_iou_thr(pred_boxes, gt_boxes, iou_threshold)
                self.correct["tp"] += correct["tp"]
                self.correct["fp"] += correct["fp"]
                self.correct["fn"] += correct["fn"]

    def reset(self) -> None:
        self.correct = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    @torch.no_grad()
    def correct_at_iou_thr(
        self, pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float
    ) -> Dict[str, int]:
        correct = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return correct

        elif len(gt_boxes) == 0:
            correct["fp"] = len(pred_boxes)
            return correct

        elif len(pred_boxes) == 0:
            correct["fn"] = len(gt_boxes)
            return correct

        iou_matrix = box_iou(pred_boxes, gt_boxes)
        num_preds, num_gt = iou_matrix.shape
        fp = torch.ones(num_gt, dtype=torch.bool)
        for ious in iou_matrix:
            iou, gt_idx = ious.max(dim=0)
            if iou >= iou_threshold:
                fp[gt_idx] = False
        fp_count = int(fp.sum())
        tp_count = num_gt - fp_count
        fn_count = num_preds - tp_count
        correct["tp"] = tp_count
        correct["fp"] = fp_count
        correct["fn"] = fn_count
        return correct
