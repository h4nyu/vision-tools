import torch.nn as nn
from typing import *
from ..modules import Conv2dStaticSamePadding
from vnet import FP


class BackboneConnector(nn.Module):
    def __init__(
        self,
        channels: int,
        conv_channels: list[int],
    ) -> None:
        super().__init__()
        assert len(conv_channels) == 5

        self.p3_conv = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[0], channels, 1),
            nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
        )
        self.p4_conv = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], channels, 1),
            nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
        )
        self.p5_conv = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], channels, 1),
            nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
        )
        self.p6_conv = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], channels, 1),
            nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
        )
        self.p7_conv = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[4], channels, 1),
            nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
        )

    def forward(self, feats: FP) -> FP:
        p3, p4, p5, p6, p7 = feats
        return (
            self.p3_conv(p3),
            self.p4_conv(p4),
            self.p5_conv(p5),
            self.p6_conv(p6),
            self.p7_conv(p7),
        )
