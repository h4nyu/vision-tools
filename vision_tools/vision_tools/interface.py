from torch import Tensor
from typing import Protocol, Callable, Any, TypedDict


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


TrainBatch = TypedDict(
    "TrainBatch",
    {
        "image_batch": Tensor,
        "box_batch": list[Tensor],
        "label_batch": list[Tensor],
    },
)

TrainSample = TypedDict(
    "TrainSample",
    {
        "id": str,
        "image": Tensor,
        "boxes": Tensor,
        "labels": Tensor,
    },
)
