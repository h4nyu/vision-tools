import torch
from torch import nn
from torch import Tensor
from typing import Callable

DefaultActivation = nn.SiLU(inplace=True)


class ConvBnAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ) -> None:
        super().__init__()
        # same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.dconv = ConvBnAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = ConvBnAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dconv(x)
        return self.pconv(x)


class SPP(nn.Module):
    """SpatialPyramidPooling"""

    def __init__(self, kernel_sizes: list[int] = [5, 9, 13]) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x] + [pool(x) for pool in self.pools], dim=1)


class ResBlock(nn.Module):
    """Standard bottleneck"""

    def __init__(
        self,
        in_channels: int,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        Conv = DWConv if depthwise else ConvBnAct
        self.conv = nn.Sequential(
            ConvBnAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                act=act,
            ),
            Conv(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                act=act,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out + x


class CSP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 1,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels // 2  # hidden channels

        self.main = nn.Sequential(
            ConvBnAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                act=act,
            ),
            *[
                ResBlock(in_channels=hidden_channels, expansion=1.0, act=act)
                for _ in range(depth)
            ]
        )
        self.bypass = ConvBnAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act=act,
        )
        self.out = ConvBnAct(
            in_channels=hidden_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )

    def forward(self, x: Tensor) -> Tensor:
        main = self.main(x)
        bypass = self.bypass(x)
        merged = torch.cat([main, bypass], dim=1)
        return self.out(merged)
