import numpy as np
import cv2
import torch
from typing import Any
from torch import Tensor
from torch.utils.data import Dataset
from object_detection.entities.image import RGB, Image
from object_detection.entities.box import (
    Boxes,
    YoloBoxes,
    pascal_to_yolo,
    filter_size,
)
from object_detection.entities import (
    Image,
    ImageSize,
    YoloBoxes,
    Labels,
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
        self.boxes = Boxes(torch.empty((0, 4), dtype=torch.int32))
        self.labels: list[int] = []

    def add_triangle(self, max_size: int = 128) -> None:
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
        self.image = cv2.polylines(self.image, pts, 1, (0, 0, 0))
        x0 = np.min(pts[:, :, 0])
        y0 = np.min(pts[:, :, 1])
        x1 = np.max(pts[:, :, 0])
        y1 = np.max(pts[:, :, 1])
        boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.int32)
        self.boxes = Boxes(torch.cat([self.boxes, boxes]))
        self.labels.append(0)

    def add_circle(self, max_size: int = 128) -> None:
        cx = np.random.randint(0, self.width - max_size)
        cy = np.random.randint(0, self.height - max_size)
        radius = random.randint(1, max_size // 2)

        self.image = cv2.circle(self.image, (cx, cy), radius, (0, 0, 0))
        x0 = np.max(cx - radius, 0)
        y0 = np.max(cy - radius, 0)
        x1 = cx + radius
        y1 = cy + radius
        boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.int32)
        self.boxes = Boxes(torch.cat([self.boxes, boxes]))
        self.labels.append(1)

    def add(self, max_size: int) -> None:
        random.choice(
            [
                self.add_triangle,
                self.add_circle,
            ]
        )(max_size)

    def __call__(self) -> tuple[Image, Boxes, Labels]:
        img = torch.from_numpy(self.image).permute(2, 0, 1) / 255  # [H, W]
        boxes = self.boxes
        labels = Labels(torch.tensor(self.labels, dtype=torch.int32))
        return Image(img.float()), boxes, labels


class TrainDataset(Dataset):
    def __init__(
        self,
        image_size: ImageSize,
        object_size_range: tuple[int, int],
        object_count_range: tuple[int, int] = (1, 10),
        num_samples: int = 100,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.object_count_range = object_count_range
        self.object_size_range = object_size_range

    def __getitem__(self, idx: int) -> tuple[str, Image, Boxes, Labels]:
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
        image, boxes, labels = poly()
        boxes, indices = filter_size(boxes, lambda x: x > 36)
        labels = Labels(labels[indices])
        return (
            "",
            Image(image),
            Boxes(boxes.float()),
            labels,
        )

    def __len__(self) -> int:
        return self.num_samples
