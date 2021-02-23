import torch
from object_detection.agc import unitwise_norm, AGC


def test_unitwise_norm() -> None:
    x = torch.tensor([[1, 1], [1, 1]])
    y = unitwise_norm(x)
    assert x.ndim == y.ndim
    assert x.shape[1:] == y.shape[1:]
