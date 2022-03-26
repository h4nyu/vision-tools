import torch
from torch import nn, optim

from hwad_bench.models import ConvNeXt, EfficientNet


def test_model() -> None:
    embedding_size = 512
    model = ConvNeXt(name="convnext_tiny", embedding_size=embedding_size)
    images = torch.randn(2, 3, 256, 256)
    output = model(images)
    assert output.shape == (2, embedding_size)


def test_eff() -> None:
    embedding_size = 512
    model = EfficientNet(name="efficientnet_b0", embedding_size=embedding_size)
    images = torch.randn(2, 3, 256, 256)
    output = model(images)
    assert output.shape == (2, embedding_size)