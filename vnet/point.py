import torch
from typing import NewType, Union
from . import Number

Points = NewType("Points", torch.Tensor)  # [B, 2] [x, y]


def resize_points(points: Points, scale_x: Number, scale_y: Number) -> Points:
    if len(points) == 0:
        return points
    x, y = points.unbind(-1)
    return Points(torch.stack([x * scale_x, y * scale_y], dim=-1))


def shift_points(points: Points, shift: tuple[Number, Number]) -> Points:
    if len(points) == 0:
        return points
    shift_x, shift_y = shift
    x, y = points.unbind(-1)
    return Points(torch.stack([x + shift_x, y + shift_y], dim=-1))
