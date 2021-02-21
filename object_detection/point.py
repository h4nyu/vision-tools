import torch
from typing import NewType, Union
from . import Number

Points = NewType("Points", torch.Tensor)  # [B, 2] [x, y]


def resize_points(points: Points, scale: tuple[Number, Number]) -> Points:
    if len(points) == 0:
        return points
    wr, hr = scale
    x, y = points.unbind(-1)
    return Points(torch.stack([x * wr, y * hr], dim=-1))
