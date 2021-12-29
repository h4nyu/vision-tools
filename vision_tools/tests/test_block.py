import torch
from vision_tools.block import ConvBnAct, DWConv


def test_conv_bn_act() -> None:
    in_channels = 32
    out_channels = 24
    b = ConvBnAct(in_channels=in_channels, out_channels=out_channels)
    inputs = torch.rand(2, in_channels, 32, 32)
    res = b(inputs)
    assert res.shape == (2, out_channels, 32, 32)


def test_dwconv() -> None:
    in_channels = 32
    out_channels = 24
    b = DWConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
    inputs = torch.rand(2, in_channels, 32, 32)
    res = b(inputs)
    assert res.shape == (2, out_channels, 32, 32)
