import typing as t
from torch import nn, Tensor
import torch.nn.functional as F
from .modules import Hswish, ConvBR2d, CSE2d, Mish, FReLU
from typing_extensions import Literal


class MobileV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: Literal[3, 5] = 3,
        stride: Literal[1, 2] = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.stride = stride
        self.is_shortcut = (stride == 1) and (in_channels == out_channels)
        self.conv = nn.Sequential(
            ConvBR2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            FReLU(mid_channels),
            ConvBR2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_channels,
                bias=False,
            ),
            CSE2d(mid_channels, reduction=4),
            FReLU(mid_channels),
            # pw-linear
            ConvBR2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):  # type: ignore
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            return x + self.conv(x)
        else:
            return self.conv(x)


class SENextBottleneck2d(nn.Module):
    pool: t.Union[None, nn.MaxPool2d, nn.AvgPool2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 8,
        groups: int = 16,
        pool: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.conv = nn.Sequential(
            ConvBR2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            FReLU(mid_channels),
            ConvBR2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            FReLU(mid_channels),
        )

        self.bypass = nn.Sequential()
        if stride > 1:
            if pool == "max":
                self.conv.add_module("pool", nn.MaxPool2d(stride, stride))
            elif pool == "avg":
                self.conv.add_module("pool", nn.AvgPool2d(stride, stride))
            self.bypass.add_module("pool", nn.AvgPool2d(stride, stride))
        self.conv.add_module(
            "conv3",
            nn.Sequential(
                ConvBR2d(
                    mid_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
            ),
        )
        if in_channels != out_channels:
            self.bypass.add_module(
                "conv0",
                ConvBR2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
            )
        self.cse = CSE2d(out_channels, reduction)
        self.activation = FReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        s = self.conv(x)
        s = self.cse(s)

        x = self.bypass(x)

        x = x + s
        x = self.activation(x)
        return x
