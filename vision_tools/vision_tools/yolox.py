import torch
from torch import nn
from torch import Tensor
from typing import Callable
import math
from .block import DefaultActivation, DWConv, ConvBnAct


class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_channels: int,
        act: Callable = DefaultActivation,
        depthwise: bool = False,
    ):
        super().__init__()
        Conv = DWConv if depthwise else ConvBnAct
        self.stem = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act=act,
        )
        self.reg_branch = nn.Sequential(
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )

        self.cls_branch = nn.Sequential(
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.cls_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.init_weights()

    def init_weights(self, prior_prob: float = 1e-2) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        for conv in [self.cls_out, self.obj_out]:
            if conv.bias is None:
                continue
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        reg_feat = self.reg_branch(stem)
        cls_feat = self.cls_branch(stem)
        reg_out = self.reg_out(reg_feat)
        obj_out = self.obj_out(reg_feat)
        cls_out = self.cls_out(cls_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class YOLOXHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hidden_channels: int,
        num_classes: int = 1,
        depthwise: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                DecoupledHead(
                    in_channels=c,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    act=act,
                )
                for c in in_channels
            ]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        return [m(x) for m, x in zip(self.heads, feats)]
