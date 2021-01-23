import torch
from typing import *
from torch import nn, Tensor
import torch.nn.functional as F
from object_detection.entities import FP
from .bottlenecks import SENextBottleneck2d
from .modules import (
    SeparableConv2d,
    Conv2dStaticSamePadding,
    MaxPool2dStaticSamePadding,
    MemoryEfficientSwish,
)


class Down2d(nn.Module):
    def __init__(
        self,
        channels: int,
        bilinear: bool = False,
        merge: bool = True,
    ) -> None:
        super().__init__()
        self.down = SENextBottleneck2d(
            in_channels=channels,
            out_channels=channels,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        return x


class Up2d(nn.Module):
    up: Union[nn.Upsample, nn.ConvTranspose2d]

    def __init__(
        self,
        channels: int,
        merge: bool = True,
    ) -> None:
        super().__init__()
        self.merge = merge
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):  # type: ignore
        x = self.up(x)
        diff_h = torch.tensor([x.size()[2] - t.size()[2]])
        diff_w = torch.tensor([x.size()[3] - t.size()[3]])
        x = F.pad(x, (diff_h - diff_h // 2, diff_w - diff_w // 2))
        return x


class Merge2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.merge = SENextBottleneck2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
        )

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat(inputs, dim=1)
        x = self.merge(x)
        return x


# class BiFPN(nn.Module):
#     def __init__(self, channels: int) -> None:
#         super().__init__()

#         self.up3 = Up2d(channels)
#         self.up4 = Up2d(channels)
#         self.up5 = Up2d(channels)
#         self.up6 = Up2d(channels)
#         self.up7 = Up2d(channels)

#         self.mid6 = Merge2d(in_channels=channels * 2, out_channels=channels)
#         self.mid5 = Merge2d(in_channels=channels * 2, out_channels=channels)
#         self.mid4 = Merge2d(in_channels=channels * 2, out_channels=channels)

#         self.out3 = Merge2d(in_channels=channels * 2, out_channels=channels)
#         self.out4 = Merge2d(in_channels=channels * 3, out_channels=channels)
#         self.out5 = Merge2d(in_channels=channels * 3, out_channels=channels)
#         self.out6 = Merge2d(in_channels=channels * 3, out_channels=channels)
#         self.out7 = Merge2d(in_channels=channels * 2, out_channels=channels)

#         self.down3 = Down2d(channels)
#         self.down4 = Down2d(channels)
#         self.down5 = Down2d(channels)
#         self.down6 = Down2d(channels)
#         self.down7 = Down2d(channels)

#     def forward(self, inputs: FP) -> FP:
#         p3_in, p4_in, p5_in, p6_in, p7_in = inputs
#         p7_up = self.up7(p7_in, p6_in)
#         p6_mid = self.mid6(p7_up, p6_in)
#         p6_up = self.up7(p6_mid, p5_in)
#         p5_mid = self.mid5(p6_up, p5_in)
#         p5_up = self.up5(p5_mid, p4_in)
#         p4_mid = self.mid4(p5_up, p4_in)
#         p4_up = self.up4(p4_mid, p3_in)

#         p3_out = self.out3(p3_in, p4_up)
#         p3_down = self.down3(p3_out)
#         p4_out = self.out4(p3_down, p4_mid, p4_in)
#         p4_down = self.down4(p4_out)
#         p5_out = self.out5(p4_down, p5_mid, p5_in)
#         p5_down = self.down5(p5_out)
#         p6_out = self.out6(p5_down, p6_mid, p6_in)
#         p6_down = self.down6(p6_out)
#         p7_out = self.out7(p6_down, p7_in)

#         return p3_out, p4_out, p5_out, p6_out, p7_out


class BiFPN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv6_up = SeparableConv2d(channels)
        self.conv5_up = SeparableConv2d(channels)
        self.conv4_up = SeparableConv2d(channels)
        self.conv3_up = SeparableConv2d(channels)
        self.conv4_down = SeparableConv2d(channels)
        self.conv5_down = SeparableConv2d(channels)
        self.conv6_down = SeparableConv2d(channels)
        self.conv7_down = SeparableConv2d(channels)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish()

        self.p6_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs: FP) -> FP:
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))
        p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))
        p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))
        p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out
