import torch
from torch import nn
from app.models.detr import DETR
from app.models.utils import NestedTensor


def test_detr() -> None:
    inputs = NestedTensor(torch.rand(2, 3, 1024, 1024), None)
    fn = DETR()
    #  fn.forward(inputs)
