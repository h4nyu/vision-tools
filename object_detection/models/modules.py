import torch, math
from typing import *
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, i: Any) -> Any:  # type: ignore
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # type: ignore
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return SwishImplementation.apply(x)


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
        kernel_size: int = 3,
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


class Conv2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            groups=groups,
        )
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        # self.stride = tuple(self.stride, self.stride)
        # elif len(self.stride) == 1:
        #     self.stride = tuple([self.stride[0]] * 2)

        # if isinstance(self.kernel_size, int):
        #     self.kernel_size = tuple([self.kernel_size] * 2)
        # elif len(self.kernel_size) == 1:
        #     self.kernel_size = tuple([self.kernel_size[0]] * 2)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]

        extra_h = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            - w
            + self.kernel_size[1]
        )
        extra_v = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            - h
            + self.kernel_size[0]
        )

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False,
        )
        if out_channels is None:
            self.pointwise_conv = Conv2dStaticSamePadding(
                in_channels, in_channels, kernel_size=1, stride=1
            )
        else:
            self.pointwise_conv = Conv2dStaticSamePadding(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SeparableConvBR2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = SeparableConv2d(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.stride = (stride, stride)
        self.kernel_size = (kernel_size, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        extra_h = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            - w
            + self.kernel_size[1]
        )
        extra_v = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            - h
            + self.kernel_size[0]
        )

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x
