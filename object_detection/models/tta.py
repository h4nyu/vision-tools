from typing import Any
from object_detection.entities import ImageBatch
from torch import nn, Tensor


class TTA:
    def __init__(self,) -> None:
        ...

    def __call__(self, model: nn.Module, images: ImageBatch) -> Any:
        ...
