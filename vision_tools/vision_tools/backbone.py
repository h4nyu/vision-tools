from typing import Callable, List

from efficientnet_pytorch import EfficientNet as _EfficientNet
from torch import Tensor, nn

from .block import CSP, SPP, ConvBnAct, DefaultActivation, DWConv, Focus

efficientnet_channels = {
    "efficientnet-b0": [3, 16, 24, 40, 112, 320, 1280],
    "efficientnet-b1": [3, 16, 24, 40, 112, 320, 1280],
    "efficientnet-b2": [3, 16, 24, 48, 120, 352, 1408],
    "efficientnet-b3": [3, 24, 32, 48, 136, 384, 1536],
    "efficientnet-b4": [3, 24, 32, 56, 160, 448, 1792],
    "efficientnet-b5": [3, 24, 40, 64, 176, 512, 2048],
    "efficientnet-b6": [3, 32, 40, 72, 200, 576, 2304],
    "efficientnet-b7": [3, 32, 48, 80, 224, 640, 2560],
}
efficientnet_strides = [1, 2, 4, 8, 16, 32, 32]


class EfficientNet(nn.Module):
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        # https://github.com/lukemelas/EfficientNet-PyTorch/issues/278
        self.net = _EfficientNet.from_pretrained(name)
        self.out_len = 6
        self.channels = efficientnet_channels[name][: self.out_len]
        self.strides = efficientnet_strides[: self.out_len]

    def forward(self, images: Tensor) -> List[Tensor]:  # P1 - P6, P7 is dropped
        features = self.net.extract_endpoints(images)
        return [images, *features.values()][: self.out_len]


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
        self.stem = Conv(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=6,
            stride=2,
            act=act,
        )
        self.darks = nn.ModuleList()
        self.channels = [in_channels, base_channels]
        self.strides = [1, 2]
        for i in range(height):
            prev_ch = base_channels * 2**i
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

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = []
        feats.append(x)
        out = self.stem(x)
        feats.append(out)
        for m in self.darks:
            out = m(out)
            feats.append(out)
        return feats
