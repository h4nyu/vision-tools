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
        num_classes: int = 1,
        reid_dim: int = 0,
        width: float = 1.0,
        depthwise: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        Conv = DWConv if depthwise else ConvBnAct
        self.stems = nn.ModuleList()

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        ...
        # outputs = []
        # for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, feats)):
        #     x = self.stems[k](x)

        #     # classify
        #     cls_feat = cls_conv(x)
        #     cls_output = self.cls_preds[k](cls_feat)

        #     # regress, object, (reid)
        #     reg_feat = reg_conv(x)
        #     reg_output = self.reg_preds[k](reg_feat)
        #     obj_output = self.obj_preds[k](reg_feat)
        #     if self.reid_dim > 0:
        #         reid_output = self.reid_preds[k](reg_feat)
        #         output = torch.cat([reg_output, obj_output, cls_output, reid_output], 1)
        #     else:
        #         output = torch.cat([reg_output, obj_output, cls_output], 1)
        #     outputs.append(output)

        # return outputs
