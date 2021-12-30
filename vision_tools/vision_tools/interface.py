from torch import Tensor
from typing import Protocol, Callable, Any


class BackboneLike(Protocol):
    channels: list[int]
    strides: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...


class FPNLike(Protocol):
    channels: list[int]
    strides: list[int]

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        ...
