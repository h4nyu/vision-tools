import torch.nn as nn
import torch
from torch import Tensor
from vnet import FP
from efficientnet_pytorch.model import EfficientNet
from typing_extensions import Literal
from typing import Any
from .connector import BackboneConnector

SIDEOUT: Any = {  # phi: (stages, channels)
    0: ([0, 2, 4, 10, 15], [16, 24, 40, 112, 320]),
    1: ([1, 4, 7, 15, 22], [16, 24, 40, 112, 320]),
    2: ([1, 4, 7, 15, 22], [16, 24, 48, 120, 352]),
    3: ([1, 4, 7, 17, 25], [24, 32, 48, 136, 384]),
    4: ([1, 5, 9, 21, 31], [24, 32, 56, 160, 448]),
    5: ([2, 7, 12, 26, 38], [24, 40, 64, 176, 512]),
    6: ([2, 8, 14, 30, 44], [32, 40, 72, 200, 576]),
    7: ([3, 10, 17, 37, 54], [32, 48, 80, 224, 640]),
}

Phi = Literal[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]


class EfficientNetBackbone(nn.Module):
    def __init__(
        self,
        phi: int,
        out_channels: int,
        pretrained: bool = False,
    ):
        super().__init__()
        model_name = f"efficientnet-b{phi}"
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model
        (
            self._sideout_stages,
            self.sideout_channels,
        ) = SIDEOUT[phi]
        self.convs = BackboneConnector(
            out_channels,
            self.sideout_channels,
        )

    def forward(self, x: torch.Tensor) -> FP:
        m = self.model
        x = m._swish(m._bn0(m._conv_stem(x)))
        feats = []
        for idx, block in enumerate(m._blocks):
            drop_connect_rate = m._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(m._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self._sideout_stages:
                feats.append(x)
        return self.convs(feats)
