import torch
from vision_tools.backbone import CSPDarknet


def test_cspdarknet() -> None:
    m = CSPDarknet()
    img = torch.rand(2, 3, 256, 256 + 64)
    feats = m(img)
    assert len(m.channels) == len(m.strides) == len(feats)
    for f, s, c in zip(feats, m.strides, m.channels):
        f.size() == (2, c, img.size(2) // s, img.size(3) // s)
