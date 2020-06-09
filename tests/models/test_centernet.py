import torch
from app.models.centernet import BoxRegression, LabelClassification, CenterNet


def test_boxregression() -> None:
    inputs = torch.rand((10, 128, 32, 32))
    fn = BoxRegression(in_channels=128, num_queries=100)
    outs = fn(inputs)
    assert outs.shape == (10, 100, 4)


def test_labelclassification() -> None:
    inputs = torch.rand((10, 128, 32, 32))
    fn = LabelClassification(in_channels=128, num_queries=100, num_classes=2)
    outs = fn(inputs)
    assert outs.shape == (10, 100, 2)


def test_centernet() -> None:
    inputs = torch.rand((10, 3, 32, 32))
    fn = CenterNet(num_queries=100, num_classes=2)
    outs = fn(inputs)
