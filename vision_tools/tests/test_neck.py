import torch

from vision_tools.neck import CSPPAFPN


def test_cspbifpn() -> None:
    size = 512
    in_channels = [
        32,
        64,
        128,
    ]
    strides = [2, 4, 8]
    neck = CSPPAFPN(in_channels=in_channels, strides=strides)
    in_feats = [
        torch.rand(2, c, size // s, size // s) for c, s in zip(in_channels, strides)
    ]
    out_feats = neck(in_feats)
    assert len(out_feats) == len(in_feats) == len(neck.channels)
    for f, c, s in zip(out_feats, neck.channels, neck.strides):
        assert f.size() == (2, c, size // s, size // s)
