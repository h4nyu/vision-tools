from typing import Tuple

import torch
from torch import Tensor

from . import Number


def resize_points(points: Tensor, scale_x: Number, scale_y: Number) -> Tensor:
    if len(points) == 0:
        return points
    x, y = points.unbind(-1)
    return torch.stack([x * scale_x, y * scale_y], dim=-1)


def shift_points(points: Tensor, shift: Tuple[Number, Number]) -> Tensor:
    if len(points) == 0:
        return points
    shift_x, shift_y = shift
    x, y = points.unbind(-1)
    return torch.stack([x + shift_x, y + shift_y], dim=-1)
