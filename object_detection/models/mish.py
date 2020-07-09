import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * (torch.tanh(F.softplus(x)))
