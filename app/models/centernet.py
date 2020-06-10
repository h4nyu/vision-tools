import torch
import numpy as np
import typing as t
import torchvision
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from .modules import ConvBR2d, SENextBottleneck2d
from .bifpn import BiFPN, FP
from app.dataset import Targets, Target
from scipy.stats import multivariate_normal


class GussiianFilter(nn.Module):
    def __init__(self, kernel_size: int, sigma: float, channels: int,) -> None:
        super().__init__()
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0
        kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.filter = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )
        self.filter.weight.data = kernel
        self.filter.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.filter(x)


class Backbone(nn.Module):
    def __init__(self, name: str, out_channels: int) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        if name == "resnet34" or name == "resnet18":
            num_channels = 512
        else:
            num_channels = 2048
        self.layers = list(self.backbone.children())[:-2]
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=i, out_channels=out_channels, kernel_size=1,)
            for i in [64, 64, 128, 256, 512]
        ])

    def forward(self, x: Tensor) -> FP:
        internal_outputs = []
        for layer in self.layers:
            x = layer(x)
            internal_outputs.append(x)
        _, p3, _, p4, _, p5, p6, p7 = internal_outputs
        return (
            self.projects[0](p3),
            self.projects[1](p4),
            self.projects[2](p5),
            self.projects[3](p6),
            self.projects[4](p7),
        )


class CenterHeatMap(nn.Module):
    def __init__(self, w: int, h: int, kernel_size: int = 3,) -> None:
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, boxes: Tensor) -> Tensor:
        img = torch.zeros((1, 3, self.w, self.h)).to(boxes.device)
        b, _ = boxes.shape
        if b == 0:
            return img
        x0 = (boxes[:, 0] * self.w).long()
        y0 = (boxes[:, 1] * self.h).long()
        img[:, 0, x0, y0] = 1
        img[:, 1, x0, y0] = boxes[:, 2]
        img[:, 2, x0, y0] = boxes[:, 3]
        return img


class PreProcess(nn.Module):
    def __init__(self,)->None:
        super().__init__()
        self.heatmap = CenterHeatMap(w=1024//2, h=1024//2)

    def forward(self, batch: t.Tuple[Tensor, t.List[Target]]) -> t.Tuple[Tensor, Tensor]:
        images, targets = batch
        heatmaps = torch.cat([
            self.heatmap(t['boxes'])
            for t
            in targets
        ], dim=0)
        return images, heatmaps


class Criterion(nn.Module):
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return F.smooth_l1_loss(src, tgt)
        #  src = src.unsqueeze(1).float()
        #  tgt = tgt.unsqueeze(1).float()
        #  pos_inds = tgt.eq(1).float()
        #  neg_inds = tgt.lt(1).float()
        #  neg_weights = torch.pow(1 - tgt, 4)
        #  oss = torch.tensor(0).to(src.device)
        #  pos_loss = torch.log(src + 1e-12) * torch.pow(1 - src, 3) * pos_inds
        #  neg_loss = (
        #      torch.log(1 - src + 1e-12) * torch.pow(src, 3) * neg_weights * neg_inds
        #  )
        #  num_pos = pos_inds.float().sum()
        #  pos_loss = pos_loss.sum()
        #  neg_loss = neg_loss.sum()
        #  loss = torch.tensor(0)
        #  if num_pos == 0:
        #      loss = loss - neg_loss
        #  else:
        #      loss = loss - (pos_loss + neg_loss) / num_pos
        #  return loss


Outputs = t.Tuple[Tensor, Tensor]


class CenterNet(nn.Module):
    def __init__(
        self, name: str = "resnet18", num_classes: int = 1, num_queries: int = 100
    ) -> None:
        super().__init__()
        channels = 64
        self.backbone = Backbone(name, out_channels=channels)
        self.fpn = BiFPN(channels=channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)
        self.outr = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, 3, W, H]
        """
        fp = self.backbone(x)
        fp = self.fpn(fp)

        outc = self.outc(fp[0])
        outr = self.outr(fp[0])
        return torch.cat([outc, outr], dim=1).sigmoid()
