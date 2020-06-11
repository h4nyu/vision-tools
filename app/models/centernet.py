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
from .utils import box_cxcywh_to_xyxy


def gaussian_2d(shape:t.Any, sigma:float=1) -> np.ndarray:
    m, n = int((shape[0] - 1.) / 2.), int((shape[1] - 1.) / 2.)
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


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
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h
        mount = gaussian_2d((64, 64), sigma=5)
        self.mount = torch.tensor(mount).view(1, 1, mount.shape[0], mount.shape[1])

    def forward(self, boxes: Tensor) -> Tensor:
        img = torch.zeros((1, 1, self.w, self.h)).float().to(boxes.device)
        b, _ = boxes.shape
        if b == 0:
            return img
        xyxy = box_cxcywh_to_xyxy(boxes)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * self.w
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * self.w
        xyxy = xyxy.long()
        sizes = xyxy[:, [2, 3]] - xyxy[:, [0, 1]]
        for box, size in zip(xyxy, sizes):
            x, y = box[0], box[1]
            w, h = size[0], size[1]
            img[:, :, x:x + w, y:y+h] += F.interpolate(self.mount, size=(w, h)).to(boxes.device) #type:ignore
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

Outputs = t.Tuple[Tensor, Tensor]

class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def neg_loss(self, src: Tensor, tgt: Tensor) -> Tensor:
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
        #  loss = loss - (pos_loss + neg_loss) / num_pos
        return F.l1_loss(src, tgt)
    def forward(self, src:Outputs, tgt:Tensor) -> Tensor:
        hmap, _ = src
        return self.neg_loss(hmap, tgt)




class CenterNet(nn.Module):
    def __init__(
        self, name: str = "resnet18", num_classes: int = 1, num_queries: int = 100
    ) -> None:
        super().__init__()
        channels = 32
        self.backbone = Backbone(name, out_channels=channels)
        self.fpn = nn.Sequential(
            BiFPN(channels=channels),
            BiFPN(channels=channels),
        )
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)
        self.outr = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Outputs:
        """
        x: [B, 3, W, H]
        """
        fp = self.backbone(x)
        fp = self.fpn(fp)

        outc = self.outc(fp[0]).sigmoid()
        outr = self.outr(fp[0])
        return outc, outr
