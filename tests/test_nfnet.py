import torch
from object_detection.nfnet import WSConv2d

def test_wsconv2d() -> None:
    in_channels = 32
    out_channels = 64
    h, w = 128, 128
    x = torch.ones(1, in_channels, h, w)
    block = WSConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
    )
    y = block(x)
    # TODO assert

