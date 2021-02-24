import torch
from torch import nn, Tensor
from torch.functional import F
from typing import Callable, Optional


class WSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        eps: float = 1e-4,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        nn.init.kaiming_normal_(self.weight)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weight(self) -> Tensor:
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[0:]))
        scale = torch.rsqrt(
            torch.max(var * fan_in, torch.tensor(self.eps).to(var.device))
        ) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input: Tensor) -> Tensor:
        weight = self.standardize_weight()
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.se_conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.se_conv1 = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        out = torch.mean(x, dim=[2, 3], keepdim=True)
        out = self.se_conv1(self.activation(self.se_conv0(out)))
        return (torch.sigmoid(out) * 2) * x


class NFBlock(nn.Module):
    conv_shortcut: Optional[WSConv2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        beta: float = 1.0,
        alpha: float = 1.0,
        expansion: float = 2.25,
        se_ratio: float = 0.5,
        group_size: Optional[int] = 8,
        activation: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.beta = beta
        self.alpha = alpha
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.group_size = group_size
        self.activation = activation

        hidden_channels = int(expansion * in_channels)
        if group_size is None:
            groups = 1
        else:
            groups = hidden_channels // group_size
            hidden_channels = int(group_size * groups)

        self.hidden_channels = hidden_channels

        self.conv1x1a = WSConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.conv3x3 = WSConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
        )
        self.conv1x1b = WSConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        if in_channels != out_channels:
            self.conv_shortcut = WSConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.conv_shortcut = None

        se_channels = max(1, int(hidden_channels * se_ratio))
        self.se = SqueezeExcite(
            in_channels=hidden_channels,
            out_channels=out_channels,
            activation=activation,
        )
        self.skip_init_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x) / self.beta
        shortcut = x
        if self.stride > 1:
            shortcut = F.avg_pool2d(shortcut, 2)
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)
        out = self.conv1x1a(out)
        out = self.conv3x3(self.activation(out))
        out = self.se(out)
        out = self.conv1x1b(self.activation(out))
        return out * self.skip_init_gain * self.alpha + shortcut
