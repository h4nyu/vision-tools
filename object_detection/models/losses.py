import typing as t
import numpy as np
import torch.nn.functional as F
from typing_extensions import Literal
from torch import Tensor
import torch
import torch.nn as nn

from typing import Dict

Reduction = Literal["none", "mean", "sum"]


class FocalLoss(nn.Module):
    """
    Modified focal loss
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        pred: 0-1 [B, C,..]
        gt: 0-1 [B, C,..]
        """
        alpha = self.alpha
        gamma = self.gamma
        eps = self.eps
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.eq(0).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        neg_loss = -(pred ** alpha) * torch.log(1 - pred) * neg_mask
        loss = (pos_loss + neg_loss).sum()
        num_pos = pos_mask.sum().float()
        return loss / num_pos
