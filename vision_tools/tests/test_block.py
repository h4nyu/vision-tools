import torch
from vision_tools.block import ConvBnAct, DWConv, SPP, ResBlock, CSP, Focus


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


def test_spp() -> None:
    kernel_sizes = [5, 9, 15]
    b = SPP(kernel_sizes=kernel_sizes)
    inputs = torch.rand(2, 3, 32, 32)
    res = b(inputs)
    assert res.size() == (2, (len(kernel_sizes) + 1) * inputs.size(1), 32, 32)


def test_resblock() -> None:
    in_channels = 32
    b = ResBlock(in_channels=in_channels)
    inputs = torch.rand(2, in_channels, 32, 32)
    res = b(inputs)
    assert res.size() == inputs.size()


def test_cps() -> None:
    in_channels = 32
    out_channels = 48
    b = CSP(in_channels=in_channels, out_channels=out_channels)
    inputs = torch.rand(2, in_channels, 32, 32)
    res = b(inputs)
    assert res.size() == (2, out_channels, 32, 32)



def test_focus() -> None:
    in_channels = 32
    out_channels = 48
    b = Focus(in_channels=in_channels, out_channels=out_channels)
    inputs = torch.rand(2, in_channels, 32, 32)
    res = b(inputs)
    assert res.size() == (2, out_channels, 16, 16)
