import numpy as np
import cv2
import torch
from typing import Any, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from object_detection.entities.image import RGB, Image
from object_detection.entities.box import (
    PascalBoxes,
    YoloBoxes,
    pascal_to_yolo,
)
from object_detection.entities import (
    TrainSample,
    Image,
    ImageSize,
    YoloBoxes,
    ImageId,
    Labels,
    PredictionSample,
)
import random


class PolyImage:
    def __init__(
        self,
        height: int = 512,
        width: int = 512,
    ) -> None:
        image = np.ones((height, width, 3), np.uint8) * 255
        self.height = height
        self.width = width
        self.image = image
        self.boxes = PascalBoxes(
            torch.empty((0, 4), dtype=torch.int32)
        )

    def _get_box(self, pts: Any) -> Tuple[int, int, int, int]:
        xs = pts[:, :, 0]
        ys = pts[:, :, 1]
        x0 = np.min(xs)
        y0 = np.min(ys)
        x1 = np.max(xs)
        y1 = np.max(ys)
        return x0, y0, x1, y1

    def add(self, max_size: int = 128) -> None:
        base_x = np.random.randint(0, self.width - max_size)
        base_y = np.random.randint(0, self.height - max_size)
        xs = np.random.randint(
            low=base_x,
            high=base_x + max_size,
            size=(1, 3, 1),
        )
        ys = np.random.randint(
            low=base_y,
            high=base_y + max_size,
            size=(1, 3, 1),
        )
        pts = np.concatenate((xs, ys), axis=2)
        self.image = cv2.fillPoly(self.image, pts, (0, 0, 0))
        boxes = torch.tensor([self._get_box(pts)], dtype=torch.int32)
        self.boxes = PascalBoxes(torch.cat([self.boxes, boxes]))

    def __call__(self) -> Tuple[Image, PascalBoxes]:
        img = (
            torch.from_numpy(self.image).permute(2, 0, 1) / 255.0
        )  # [H, W]
        boxes = self.boxes
        return Image(img.float()), boxes


class TrainDataset(Dataset):
    def __init__(
        self,
        image_size: ImageSize,
        object_size_range: Tuple[int, int],
        object_count_range: Tuple[int, int] = (1, 10),
        num_samples: int = 100,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.object_count_range = object_count_range
        self.object_size_range = object_size_range

    def __getitem__(self, idx: int) -> TrainSample:
        poly = PolyImage(
            width=self.image_size[0],
            height=self.image_size[1],
        )
        count = np.random.randint(
            low=self.object_count_range[0],
            high=self.object_count_range[1],
        )
        sizes = np.random.randint(
            low=self.object_size_range[0],
            high=self.object_size_range[1],
            size=(count,),
        )
        for s in sizes:
            poly.add(s)
        image, boxes = poly()
        labels = torch.zeros((len(boxes),), dtype=torch.int32)
        return (
            ImageId(""),
            Image(image),
            PascalBoxes(boxes.float()),
            Labels(labels),
        )

    def __len__(self) -> int:
        return self.num_samples


class PredictionDataset(Dataset):
    def __init__(
        self,
        image_size: ImageSize,
        object_size_range: Tuple[int, int],
        object_count_range: Tuple[int, int] = (1, 10),
        num_samples: int = 100,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.object_count_range = object_count_range
        self.object_size_range = object_size_range

    def __getitem__(self, idx: int) -> PredictionSample:
        poly = PolyImage(
            width=self.image_size[0],
            height=self.image_size[1],
        )
        count = np.random.randint(
            low=self.object_count_range[0],
            high=self.object_count_range[1],
        )
        sizes = np.random.randint(
            low=self.object_size_range[0],
            high=self.object_size_range[1],
            size=(count,),
        )
        for s in sizes:
            poly.add(s)
        image, _ = poly()
        return (
            ImageId(""),
            Image(image),
        )

    def __len__(self) -> int:
        return self.num_samples
