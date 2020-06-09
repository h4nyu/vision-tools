import typing as t
from torch import nn, Tensor
import torch.nn.functional as F


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
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class ConvBR2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
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


class SENextBottleneck2d(nn.Module):
    pool: t.Union[None, nn.MaxPool2d, nn.AvgPool2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 8,
        groups: int = 16,
        pool: t.Literal["max", "avg"] = "max",
        is_shortcut: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.conv1 = ConvBR2d(
            in_channels, mid_channels, kernel_size=1, padding=0, stride=1,
        )
        self.conv2 = ConvBR2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=groups,
        )
        self.conv3 = ConvBR2d(
            mid_channels, out_channels, kernel_size=1, padding=0, stride=1
        )
        self.se = CSE2d(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        self.act = nn.ReLU(inplace=True)
        if self.is_shortcut:
            self.shortcut = ConvBR2d(in_channels, out_channels, 1, 0, 1)
        if stride > 1:
            if pool == "max":
                self.pool = nn.MaxPool2d(stride, stride)
            elif pool == "avg":
                self.pool = nn.AvgPool2d(stride, stride)

    def forward(self, x: Tensor) -> Tensor:
        s = self.conv1(x)
        s = self.act(s)
        s = self.conv2(s)
        s = self.act(s)
        if self.stride > 1 and self.pool is not None:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)
        x = x + s
        x = self.act(x)
        return x
