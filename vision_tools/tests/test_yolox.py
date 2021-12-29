import pytest
import torch
from vision_tools.yolox import YOLOXHead, DecoupledHead


def test_head() -> None:
    in_channels = [
        32,
        64,
        128,
    ]
    head = YOLOXHead(
        in_channels=in_channels,
    )
    feats = [torch.rand(2, c, 32, 32) for c in in_channels]
    res = head(feats)


@pytest.mark.parametrize("depthwise", [True, False])
def test_decoupled_head(depthwise: bool) -> None:
    in_channels = 32
    hidden_channels = 48
    num_classes = 3
    head = DecoupledHead(
        num_classes=num_classes,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        depthwise=depthwise,
    )
    feat = torch.rand(2, in_channels, 32, 32)
    res = head(feat)
    assert res.size() == (2, 4 + 1 + num_classes, 32, 32)
