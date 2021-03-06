import torch
from vnet.bottlenecks import (
    MobileV3,
    SENextBottleneck2d,
)


def test_mobilev3() -> None:
    req = torch.ones((1, 32, 10, 10))
    m = MobileV3(in_channels=32, out_channels=16, mid_channels=32)
    res = m(req)
    assert res.shape == (1, 16, 10, 10)

    m = MobileV3(in_channels=32, out_channels=32, mid_channels=64)
    res = m(req)
    assert res.shape == (1, 32, 10, 10)

    m = MobileV3(
        in_channels=32,
        out_channels=32,
        mid_channels=64,
        stride=2,
    )
    res = m(req)
    assert res.shape == (1, 32, 5, 5)


def test_senext() -> None:
    inputs = torch.rand((1, 32, 10, 10))
    fn = SENextBottleneck2d(in_channels=32, out_channels=32)
    outputs = fn(inputs)
    assert inputs.shape == outputs.shape

    fn = SENextBottleneck2d(in_channels=32, out_channels=32, stride=2)
    outputs = fn(inputs)
    assert inputs.shape[:2] == outputs.shape[:2]
