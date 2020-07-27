import torch
from torch import nn, Tensor


class FReLU(nn.Module):
    def __init__(self, in_channels: int, kerel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kerel_size,
            stride=1,
            padding=kerel_size // 2,
            groups=in_channels,
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.conv(x)
        x0 = self.bn(x)
        x = torch.max(x, x0)
        return x
