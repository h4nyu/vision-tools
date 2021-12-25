import torch
from typing import *
from torch import nn, Tensor
import torch.nn.functional as F
from vision_tools import FP
from .bottlenecks import SENextBottleneck2d
from .modules import (
    SeparableConvBR2d,
    Conv2dStaticSamePadding,
    MaxPool2dStaticSamePadding,
    MemoryEfficientSwish,
)


class BiFPN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv6_up = SeparableConvBR2d(channels)
        self.conv5_up = SeparableConvBR2d(channels)
        self.conv4_up = SeparableConvBR2d(channels)
        self.conv3_up = SeparableConvBR2d(channels)
        self.conv4_down = SeparableConvBR2d(channels)
        self.conv5_down = SeparableConvBR2d(channels)
        self.conv6_down = SeparableConvBR2d(channels)
        self.conv7_down = SeparableConvBR2d(channels)

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
