import typing as t
import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import Literal
from torch import Tensor
from object_detection.entities import PascalBoxes
from torchvision.ops.boxes import box_area
import torch
import torch.nn as nn

from typing import Dict, Tuple

Reduction = Literal["none", "mean", "sum"]


class HuberLoss:
    def __init__(
        self, size_average: bool = True, delta: float = 1.0
    ) -> None:
        self.delta = delta
        self.size_average = size_average

    def __call__(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        err = src - tgt
        abs_err = err.abs()
        quadratic = torch.clamp(abs_err, max=self.delta)
        linear = abs_err - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean() if self.size_average else loss.sum()


class SigmoidFocalLoss:
    def __init__(
        self,
        gamma: float = 2.0,
        size_average: bool = True,
        alpha: float = 0.25,
    ):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        n_classes = target.shape[1]
        class_ids = torch.arange(
            1, n_classes + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)
        p = torch.sigmoid(source)
        gamma = self.gamma
        alpha = self.alpha
        pos = (1 - p) ** gamma * torch.log(p)
        neg = p ** gamma * torch.log(1 - p)
        loss = (
            -(target == class_ids).float() * alpha * pos
            - ((target != class_ids) * target >= 0).float()
            * (1 - alpha)
            * neg
        )
        return loss


class FocalLoss:
    def __init__(
        self,
        gamma: float = 2.0,
        eps: float = 1e-4,
        alpha: float = 0.25,
        size_average: bool = True,
    ):
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        pred:
            value_range: 0-1
            shape: [B, C,..]
        gt:
            0,1 [B, C,..]
        """
        gamma = self.gamma
        eps = self.eps
        alpha = self.alpha
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_loss = (
            -((1 - pred) ** gamma) * torch.log(pred) * gt.eq(1.0)
        )
        neg_loss = -(pred ** gamma) * torch.log(1 - pred) * gt.eq(0.0)
        loss = alpha * pos_loss + (1 - alpha) * neg_loss
        return loss


class IoU:
    def __call__(
        self, boxes1: PascalBoxes, boxes2: PascalBoxes
    ) -> t.Tuple[Tensor, Tensor]:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


class GIoU:
    def __init__(self) -> None:
        self.iou = IoU()

    def __call__(self, src: PascalBoxes, tgt: PascalBoxes) -> Tensor:
        iou, union = self.iou(src, tgt)
        lt = torch.min(src[:, None, :2], tgt[:, :2])
        rb = torch.max(src[:, None, 2:], tgt[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]
        return 1 - iou + (area - union) / area


class DIoU:
    def __init__(self) -> None:
        self.iou = IoU()

    def __call__(self, src: PascalBoxes, tgt: PascalBoxes) -> Tensor:
        iou, _ = self.iou(src, tgt)
        s_ctr = (src[:, None, :2] + src[:, None, 2:]) / 2
        t_ctr = (tgt[:, :2] + tgt[:, 2:]) / 2

        lt = torch.min(src[:, None, :2], tgt[:, :2])
        rb = torch.max(src[:, None, 2:], tgt[:, 2:])

        diagnol = torch.pow((rb - lt).clamp(min=0), 2).sum(dim=-1)
        ctr_dist = torch.pow(s_ctr - t_ctr, 2).sum(dim=-1)

        return 1 - iou + ctr_dist / diagnol


class IoULoss:
    def __init__(self, size_average: bool = True) -> None:
        self.size_average = size_average

    def __call__(
        self, boxes1: PascalBoxes, boxes2: PascalBoxes
    ) -> t.Tuple[Tensor, Tensor]:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, 0] * wh[:, 1]  # [N,M]

        union = area1 + area2 - inter

        iou = inter / union
        if self.size_average:
            iou = iou.mean()
        return 1 - iou, union


class DIoULoss:
    def __init__(self, size_average: bool = True) -> None:
        self.iouloss = IoULoss(size_average=size_average)
        self.size_average = size_average

    def __call__(self, src: PascalBoxes, tgt: PascalBoxes) -> Tensor:
        iouloss, _ = self.iouloss(src, tgt)
        s_ctr = (src[:, :2] + src[:, 2:]) / 2
        t_ctr = (tgt[:, :2] + tgt[:, 2:]) / 2

        lt = torch.min(src[:, :2], tgt[:, :2])
        rb = torch.max(src[:, 2:], tgt[:, 2:])

        ctr_loss = torch.pow(s_ctr - t_ctr, 2).sum(
            dim=-1
        ) / torch.pow((rb - lt).clamp(min=0), 2).sum(dim=-1)
        if self.size_average:
            ctr_loss = ctr_loss.mean()

        return iouloss + ctr_loss
