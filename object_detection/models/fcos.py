import torch
from torch import nn
from typing import Any, List
from object_detection.entities import YoloBoxes, PascalBoxes


def centerness(boxes: torch.Tensor) -> torch.Tensor:
    left_right = boxes[:, [0, 2]]
    top_bottom = boxes[:, [1, 3]]
    return torch.sqrt(
        (left_right.min(-1)[0] / left_right.max(-1)[0])
        * (top_bottom.min(-1)[0] / top_bottom.max(-1)[0])
    )


class ToBoxes:
    def __init__(
        self,
        threshold: float,
        top_n: int,
        nms_thresold: float,
        post_top_n: int,
        min_size: float,
        n_classes: int,
    ) -> None:
        self.threshold = threshold
        self.top_n = top_n
        self.nms_thresold = nms_thresold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_classes = n_classes

    def __call__(
        self, cls_batch: List[Any], box_batch: List[Any], center_batch: List[Any],
    ) -> List[YoloBoxes]:
        ...


class Criterion:
    def __init__(self, sizes: Any, gamma: float, alpha: float,) -> None:
        self.sizes = sizes
        self.gamma = gamma
        self.alpha = alpha

    def get_sample_region(
        self,
        gt_boxes: PascalBoxes,
        strides: List[int],
        point_per_level: int,
        xs: torch.Tensor,
        ys: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        ...
