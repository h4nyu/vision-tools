from torch import Tensor
from typing import Protocol, Callable, Any


class FPNLike(Protocol):
    channels: list[int]
    strides: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...
