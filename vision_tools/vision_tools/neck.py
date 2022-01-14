import torch
from torch import nn, Tensor
from typing import Callable, List
from .block import DefaultActivation, ConvBnAct, CSP


class CSPPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        strides: List[int],
        depth: int = 3,
        act: Callable[[Tensor], Tensor] = nn.LeakyReLU(inplace=True),
    ):
        super().__init__()
        self.strides = strides
        self.channels = in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_in_convs = nn.ModuleList()
        self.up_out_convs = nn.ModuleList()

        self.down_in_convs = nn.ModuleList()
        self.down_out_convs = nn.ModuleList()

        channel_legth = len(in_channels)
        for i in range(1, channel_legth):
            next_in_c = in_channels[-i - 1]
            c = in_channels[-i]
            self.up_in_convs.append(
                ConvBnAct(
                    in_channels=c,
                    out_channels=next_in_c,
                    kernel_size=1,
                    act=act,
                )
            )
            self.up_out_convs.append(
                CSP(
                    in_channels=2 * next_in_c,
                    out_channels=next_in_c,
                    depth=depth,
                    act=act,
                )
            )

        for i in range(1, channel_legth):
            next_in_c = in_channels[i]
            c = in_channels[i - 1]
            self.down_in_convs.append(
                ConvBnAct(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=3,
                    stride=2,
                    act=act,
                )
            )

            self.down_out_convs.append(
                CSP(
                    in_channels=c + next_in_c,
                    out_channels=next_in_c,
                    depth=depth,
                    act=act,
                )
            )

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        reverted_feats = feats[::-1]
        up_outs = []
        up_out = feats[-1]
        up_outs.append(up_out)
        for i in range(len(feats) - 1):
            up_out = self.up_out_convs[i](
                torch.cat(
                    [self.upsample(self.up_in_convs[i](up_out)), feats[-i - 2]], dim=1
                )
            )
            up_outs.append(up_out)

        down_outs = []
        down_out = up_outs[-1]
        down_outs.append(down_out)
        for i in range(len(feats) - 1):
            down_out = self.down_out_convs[i](
                torch.cat([self.down_in_convs[i](down_out), up_outs[-i - 2]], dim=1)
            )
            down_outs.append(down_out)
        return down_outs
