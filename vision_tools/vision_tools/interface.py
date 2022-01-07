from torch import Tensor
from typing import Protocol, Callable, Any, TypedDict, TypeVar, Generic


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


class MeterLike(Protocol):
    @property
    def value(self) -> dict[str, float]:
        ...

    def accumulate(self, log: Any) -> None:
        ...

    def reset(self) -> None:
        ...


B = TypeVar("B", contravariant=True)


class MetricLike(Protocol[B]):
    @property
    def value(self) -> tuple[float, dict[str, float]]:
        ...

    def accumulate(self, pred: B, gt: B) -> None:
        ...

    def reset(self) -> None:
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