import torch
from vision_tools.nfnet import WSConv2d, NFBlock


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


def test_nfblock_same_size() -> None:
    channels = 64
    h, w = 128, 128
    block = NFBlock(
        in_channels=channels,
        out_channels=channels,
    )
    x = torch.ones(1, channels, h, w)
    out = block(x)
    assert x.shape == out.shape


def test_nfblock_downsize() -> None:
    channels = 64
    h, w = 128, 128
    block = NFBlock(
        in_channels=channels,
        out_channels=channels,
        stride=2,
    )
    x = torch.ones(1, channels, h, w)
    out = block(x)
    assert x.shape[:2] == out.shape[:2]
    assert x.shape[2] // 2 == out.shape[2]
    assert x.shape[3] // 2 == out.shape[3]


def test_nfblock_different() -> None:
    in_channels = 64
    out_channels = 48
    h, w = 128, 128
    block = NFBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    )
    x = torch.ones(1, in_channels, h, w)
    out = block(x)
    assert x.shape[2:] == out.shape[2:]
    assert x.shape[0] == out.shape[0]
    assert out.shape[1] == out_channels


def test_nfblock_different_downsize() -> None:
    in_channels = 64
    out_channels = 48
    h, w = 128, 128
    block = NFBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
    )
    x = torch.ones(1, in_channels, h, w)
    out = block(x)
    assert x.shape[0] == out.shape[0]
    assert out.shape[1] == out_channels
    assert x.shape[2] // 2 == out.shape[2]
    assert x.shape[3] // 2 == out.shape[3]


def test_nfnet() -> None:
    ...
