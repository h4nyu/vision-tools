import pytest
import torch

from vision_tools.backbone import CSPDarknet, EfficientNet, efficientnet_channels


def test_cspdarknet() -> None:
    m = CSPDarknet()
    img = torch.rand(2, 3, 256, 256 + 64)
    feats = m(img)
    assert len(m.channels) == len(m.strides) == len(feats)
    assert m.strides == [1, 2, 4, 8, 16, 32]
    for f, s, c in zip(feats, m.strides, m.channels):
        f.size() == (2, c, img.size(2) // s, img.size(3) // s)


@pytest.mark.parametrize("name", list(efficientnet_channels.keys()))
def test_efficient(name: str) -> None:
    size = 512
    img = torch.rand(2, 3, size, size)
    backbone = EfficientNet(name)
    features = backbone(img)
    expand_len = 6
    expected_sizes = [size // s for s in backbone.strides]
    assert (
        len(features) == expand_len == len(backbone.channels) == len(backbone.strides)
    )
    for f, s, c in zip(features, expected_sizes, backbone.channels):
        f.size() == (2, c, img.size(2) // s, img.size(3) // s)
