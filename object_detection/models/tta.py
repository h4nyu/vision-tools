import torch
from functools import partial
from torch import nn
from typing import Tuple, List, Callable
from object_detection.entities import (
    YoloBoxes,
    Confidences,
    ImageBatch,
    yolo_hflip,
    yolo_vflip,
)


class HFlipTTA:
    def __init__(
        self,
        to_boxes: Callable,
    ) -> None:
        self.img_transform = partial(torch.flip, dims=(3,))
        self.to_boxes = to_boxes
        self.box_transform = yolo_hflip

    def __call__(
        self, model: nn.Module, images: ImageBatch
    ) -> Tuple[List[YoloBoxes], List[Confidences]]:
        images = ImageBatch(self.img_transform(images))
        outputs = model(images)
        box_batch, conf_batch = self.to_boxes(outputs)
        box_batch = [self.box_transform(boxes) for boxes in box_batch]
        return box_batch, conf_batch


class VHFlipTTA:
    def __init__(
        self,
        to_boxes: Callable,
    ) -> None:
        self.img_transform = partial(torch.flip, dims=(2, 3))
        self.to_boxes = to_boxes
        self.box_transform = lambda x: yolo_vflip(yolo_hflip(x))

    def __call__(
        self, model: nn.Module, images: ImageBatch
    ) -> Tuple[List[YoloBoxes], List[Confidences]]:
        images = ImageBatch(self.img_transform(images))
        outputs = model(images)
        box_batch, conf_batch = self.to_boxes(outputs)
        box_batch = [
            self.box_transform(boxes) for boxes in box_batch
        ]  # type:ignore
        return box_batch, conf_batch


class VFlipTTA:
    def __init__(
        self,
        to_boxes: Callable,
    ) -> None:
        self.img_transform = partial(torch.flip, dims=(2,))
        self.to_boxes = to_boxes
        self.box_transform = yolo_vflip

    def __call__(
        self, model: nn.Module, images: ImageBatch
    ) -> Tuple[List[YoloBoxes], List[Confidences]]:
        images = ImageBatch(self.img_transform(images))
        outputs = model(images)
        box_batch, conf_batch = self.to_boxes(outputs)
        box_batch = [self.box_transform(boxes) for boxes in box_batch]
        return box_batch, conf_batch
