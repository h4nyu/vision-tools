import torch

from vision_tools.bifpn import BiFPN


def test_bifpn() -> None:
    req = tuple(
        torch.ones((1, 128, 1024 // (2**i), 1024 // (2**i))) for i in range(3, 8)
    )
    m = BiFPN(channels=128)
    res = m(req)

    for i, o in zip(req, res):
        assert i.shape == o.shape
