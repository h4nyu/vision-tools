import typing as t
import numpy as np
import torch.nn.functional as F
from typing_extensions import Literal
from torch import Tensor
from object_detection.entities import PascalBoxes
from torchvision.ops.boxes import box_area
import torch
import torch.nn as nn

from typing import Dict, Tuple

Reduction = Literal["none", "mean", "sum"]


class FocalLoss(nn.Module):
    """
    Modified focal loss
    """

    def __init__(
        self, gamma: float = 2.0, eps: float = 1e-4,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        pred: 0-1 [B, C,..]
        gt: 0-1 [B, C,..]
        """
        gamma = self.gamma
        eps = self.eps
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_loss = -((1 - pred) ** gamma) * torch.log(pred)
        neg_loss = -(pred ** gamma) * torch.log(1 - pred)
        loss = pos_loss + neg_loss
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
        self.box_iou = IoU()

    def __call__(self, src: PascalBoxes, tgt: PascalBoxes) -> Tensor:
        iou, union = self.box_iou(src, tgt)
        lt = torch.min(src[:, None, :2], tgt[:, :2])
        rb = torch.max(src[:, None, 2:], tgt[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]
        return iou - (area - union) / area


class BIoU:
    def __call__(self, src: PascalBoxes, tgt: PascalBoxes) -> None:

        print(src)
