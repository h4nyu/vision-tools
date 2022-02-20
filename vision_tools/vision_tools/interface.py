from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar, Union

from torch import Tensor
from typing_extensions import Protocol, TypedDict


class BackboneLike(Protocol):
    channels: List[int]
    strides: List[int]

    def __call__(self, x: Tensor) -> List[Tensor]:
        ...


class FPNLike(Protocol):
    channels: List[int]
    strides: List[int]

    def __call__(self, x: List[Tensor]) -> List[Tensor]:
        ...


class MeterLike(Protocol):
    @property
    def value(self) -> Dict[str, Union[float, int]]:
        ...

    def update(self, log: Any) -> None:
        ...

    def reset(self) -> None:
        ...


B = TypeVar("B", contravariant=True)


class MetricLike(Protocol[B]):
    @property
    def value(self) -> Tuple[float, Dict[str, Any]]:
        ...

    def update(self, pred: B, gt: B) -> None:
        ...

    def reset(self) -> None:
        ...


TrainBatch = TypedDict(
    "TrainBatch",
    {
        "image_batch": Tensor,
        "box_batch": List[Tensor],
        "label_batch": List[Tensor],
        "conf_batch": List[Tensor],
    },
)

Detection = TypedDict(
    "Detection",
    {
        "image": Tensor,
        "boxes": Tensor,
        "labels": Tensor,
        "confs": Tensor,
    },
)

Classification = TypedDict(
    "Classification",
    {
        "image": Tensor,
        "labels": Tensor,
    },
)
