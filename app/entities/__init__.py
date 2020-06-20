import numpy as np
import typing as t
from pathlib import Path
from app import config
import torch
from torch import Tensor
from .box import Boxes


class Annotation:
    id: str
    width: int
    height: int
    bboxes: Tensor  # [L, 4] 0~1
    source: str
    confidences: Tensor

    def __init__(
        self,
        id: str,
        width: int,
        height: int,
        boxes: Tensor,
        source: str = "",
        confidences: t.Optional[Tensor] = None,
    ) -> None:
        #  boxes[:, [0, 2]] = boxes[:, [0, 2]] / width
        #  boxes[:, [1, 3]] = boxes[:, [1, 3]] / height
        self.id = id
        self.width = width
        self.height = height
        self.boxes = boxes
        self.source = source
        self.confidences = (
            confidences if confidences is not None else torch.ones((boxes.shape[0],))
        )

    def __repr__(self,) -> str:
        return f"<Image id={self.id}>"


class Images:
    images: Tensor
    ids: t.List[str]

    def __init__(self, ids: t.List[str], images: Tensor) -> None:
        self.images = images
        self.ids = ids

Annotations = t.List[Boxes]
