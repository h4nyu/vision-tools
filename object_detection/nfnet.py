import torch
from torch import nn, Tensor
from torch.functional import F
from typing import Callable, Optional, TypedDict

NFNetParams = TypedDict(
    "NFNetParams",
    {
        "width": list[int],
        "depth": list[int],
        "train_imsize": int,
        "test_imsize": int,
        "RA_level": str,
        "drop_rate": float,
    },
)
nfnet_params: dict[str, NFNetParams] = {
    "F0": {
        "width": [256, 512, 1536, 1536],
        "depth": [1, 2, 6, 3],
        "train_imsize": 192,
        "test_imsize": 256,
        "RA_level": "405",
        "drop_rate": 0.2,
    },
    "F1": {
        "width": [256, 512, 1536, 1536],
        "depth": [2, 4, 12, 6],
        "train_imsize": 224,
        "test_imsize": 320,
        "RA_level": "410",
        "drop_rate": 0.3,
    },
    "F2": {
        "width": [256, 512, 1536, 1536],
        "depth": [3, 6, 18, 9],
        "train_imsize": 256,
        "test_imsize": 352,
        "RA_level": "410",
        "drop_rate": 0.4,
    },
    "F3": {
        "width": [256, 512, 1536, 1536],
        "depth": [4, 8, 24, 12],
        "train_imsize": 320,
        "test_imsize": 416,
        "RA_level": "415",
        "drop_rate": 0.4,
    },
    "F4": {
        "width": [256, 512, 1536, 1536],
        "depth": [5, 10, 30, 15],
        "train_imsize": 384,
        "test_imsize": 512,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F5": {
        "width": [256, 512, 1536, 1536],
        "depth": [6, 12, 36, 18],
        "train_imsize": 416,
        "test_imsize": 544,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F6": {
        "width": [256, 512, 1536, 1536],
        "depth": [7, 14, 42, 21],
        "train_imsize": 448,
        "test_imsize": 576,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F7": {
        "width": [256, 512, 1536, 1536],
        "depth": [8, 16, 48, 24],
        "train_imsize": 480,
        "test_imsize": 608,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
}


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
        activation: Callable[[Tensor], Tensor],
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, int(in_channels * ratio))
        self.se_conv0 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=True
        )
        self.se_conv1 = nn.Conv2d(
            hidden_channels, in_channels, kernel_size=1, bias=True
        )
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
        alpha: float = 0.2,
        expansion: float = 2.25,
        se_ratio: float = 0.5,
        group_size: int = 1,
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

        width = int(expansion * in_channels)
        self.groups = width // group_size
        self.width = int(group_size * self.groups)

        self.conv1x1a = WSConv2d(
            in_channels=self.in_channels,
            out_channels=self.width,
            kernel_size=1,
            padding=0,
        )
        self.conv3x3a = WSConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
        )
        self.conv3x3b = WSConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.groups,
        )
        self.conv1x1b = WSConv2d(
            in_channels=width,
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

        self.se = SqueezeExcite(
            in_channels=out_channels,
            ratio=se_ratio,
            activation=activation,
        )
        self.skip_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x) / self.beta
        shortcut = x
        if self.stride > 1:
            shortcut = F.avg_pool2d(shortcut, 2)
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)
        out = self.conv1x1a(out)
        out = self.conv3x3a(self.activation(out))
        out = self.conv3x3b(self.activation(out))
        out = self.conv1x1b(self.activation(out))
        out = self.se(out)
        return out * self.skip_gain * self.alpha + shortcut


class Stem(nn.Module):
    def __init__(self, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.activation = activation
        self.conv0 = WSConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.conv1 = WSConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = WSConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = WSConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.conv0(x))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.conv3(out)
        return out


class NFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        activation: Callable[[Tensor], Tensor],
        stochdepth_rate: float,
        variant: str = "F0",
        alpha: float = 0.2,
        se_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        block_params = nfnet_params[variant]

        self.train_imsize = block_params["train_imsize"]
        self.test_imsize = block_params["test_imsize"]
        self.activation = activation
        self.drop_rate = block_params["drop_rate"]

        self.stem = Stem(activation=activation)

        num_blocks, index = sum(block_params["depth"]), 0

        blocks: list[NFBlock] = []
        expected_std = 1.0
        in_channels = block_params["width"][0] // 2

        block_args = zip(
            block_params["width"],
            block_params["depth"],
            [0.5] * 4,  # bottleneck pattern
            [128] * 4,  # group pattern. Original groups [128] * 4
            [1, 2, 2, 2],  # stride pattern
        )

        for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1.0 / expected_std

                block_sd_rate = stochdepth_rate * index / num_blocks
                out_channels = block_width

                # blocks.append(
                #     NFBlock(
                #         in_channels=in_channels,
                #         out_channels=out_channels,
                #         stride=stride if block_index == 0 else 1,
                #         alpha=alpha,
                #         beta=beta,
                #         se_ratio=se_ratio,
                #         group_size=group_size,
                #         stochdepth_rate=block_sd_rate,
                #         activation=activation,
                #     )
                # )

                # in_channels = out_channels
                # index += 1

                # if block_index == 0:
                #     expected_std = 1.0

                # expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5
