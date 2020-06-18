import numpy as np
import typing as t
from skimage.io import imread
from pathlib import Path
from app import config
import torch
from torch import Tensor


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
        id = self.id
        return f"<Image {id=}>"

    def get_img(self) -> Tensor:
        image_path = Path(config.image_dir).joinpath(f"{self.id}.jpg")
        return (imread(image_path) / 255).astype(np.float32)


Annotations = t.Dict[str, Annotation]
