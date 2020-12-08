import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from typing import Dict
from object_detection.entities import FP, SideChannels
from typing_extensions import Literal

ModelName = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

SIDEOUT: Dict[ModelName, SideChannels] = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "resnet101": (64, 256, 512, 1024, 2048),
    "resnet152": (64, 256, 512, 1024, 2048),
}


class ResNetBackbone(nn.Module):
    def __init__(self, name: ModelName, out_channels: int) -> None:
        super().__init__()
        self.name = name
        self.backbone = getattr(torchvision.models, name)(pretrained=True)
        self.layers = list(self.backbone.children())[:-2]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=i,
                    out_channels=out_channels,
                    kernel_size=1,
                )
                for i in SIDEOUT[name]
            ]
        )

    @property
    def size_channels(self) -> SideChannels:
        return SIDEOUT[self.name]

    def forward(self, x: Tensor) -> FP:
        internal_outputs = []
        for layer in self.layers:
            x = layer(x)
            internal_outputs.append(x)
        _, _, p3, _, p4, p5, p6, p7 = internal_outputs
        return (
            self.projects[0](p3),
            self.projects[1](p4),
            self.projects[2](p5),
            self.projects[3](p6),
            self.projects[4](p7),
        )
