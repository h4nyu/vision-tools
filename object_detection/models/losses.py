import typing as t
import numpy as np
import torch.nn.functional as F
from typing_extensions import Literal
from torch import Tensor
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
