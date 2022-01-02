from torch import nn, Tensor
from typing import Callable
from .block import DefaultActivation, DWConv, ConvBnAct, Focus, CSP, SPP


class CSPDarknet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        depth: int = 3,
        height: int = 4,
        depthwise: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        Conv = DWConv if depthwise else ConvBnAct
        base_channels = hidden_channels
        self.stem = Focus(
            in_channels=in_channels, out_channels=base_channels, kernel_size=3, act=act
        )
        self.darks = nn.ModuleList()
        self.channels = [in_channels, base_channels]
        self.strides = [1, 2]
        for i in range(height):
            prev_ch = base_channels * 2 ** i
            next_ch = base_channels * 2 ** (i + 1)
            blocks = (
                [
                    Conv(
                        in_channels=prev_ch,
                        out_channels=next_ch,
                        kernel_size=3,
                        stride=2,
                        act=act,
                    ),
                    SPP(
                        in_channels=next_ch,
                        out_channels=next_ch,
                    ),
                ]
                if i == height - 1
                else [
                    Conv(
                        in_channels=prev_ch,
                        out_channels=next_ch,
                        kernel_size=3,
                        stride=2,
                        act=act,
                    ),
                ]
            )
            blocks.append(
                CSP(
                    in_channels=next_ch,
                    out_channels=next_ch,
                    depth=depth,
                ),
            )
            self.darks.append(nn.Sequential(*blocks))

            self.channels.append(next_ch)
            self.strides.append(2 ** (i + 2))

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = []
        feats.append(x)
        out = self.stem(x)
        feats.append(out)
        for m in self.darks:
            out = m(out)
            feats.append(out)
        return feats
