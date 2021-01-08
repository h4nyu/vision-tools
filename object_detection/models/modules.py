import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F


class FReLU(nn.Module):
    def __init__(self, in_channels: int, kerel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kerel_size,
            stride=1,
            padding=kerel_size // 2,
            groups=in_channels,
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.conv(x)
        x0 = self.bn(x)
        x = torch.max(x, x0)
        return x

class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):  # type:ignore
        return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):  # type: ignore
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):  # type:ignore
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class CSE2d(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            Mish(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            Hsigmoid(inplace=True),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class ConvBR2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x
