import numpy as np, cv2, torch
from torch import Tensor
from torch.utils.data import Dataset
from vision_tools import (
    pascal_to_yolo,
    filter_size,
    RGB,
    to_center_points,
    resize_points,
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
        self.boxes = torch.empty((0, 4), dtype=torch.int32)
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
        self.boxes = torch.cat([self.boxes, boxes])
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
        self.boxes = torch.cat([self.boxes, boxes])
        self.labels.append(1)

    def add(self, max_size: int) -> None:
        random.choice(
            [
                self.add_triangle,
                self.add_circle,
            ]
        )(max_size)

    def __call__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        img = torch.from_numpy(self.image).permute(2, 0, 1) / 255  # [H, W]
        boxes = self.boxes
        points = to_center_points(boxes)
        points = resize_points(points, scale_x=1 / self.width, scale_y=1 / self.height)
        labels = torch.tensor(self.labels, dtype=torch.int32)
        return img.float(), boxes, points, labels


class PointDataset(Dataset):
    def __init__(
        self,
        image_size: tuple[int, int],
        object_size_range: tuple[int, int],
        object_count_range: tuple[int, int] = (1, 10),
        num_samples: int = 100,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.object_count_range = object_count_range
        self.object_size_range = object_size_range

    def __getitem__(self, idx: int) -> tuple[str, Tensor, Tensor, Tensor]:
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
        image, boxes, points, labels = poly()
        _, indices = filter_size(boxes, lambda x: x > 36)
        points = points[indices]
        labels = labels[indices]
        return (
            "",
            image,
            points,
            labels,
        )

    def __len__(self) -> int:
        return self.num_samples


class BoxDataset(Dataset):
    def __init__(
        self,
        image_size: tuple[int, int],
        object_size_range: tuple[int, int],
        object_count_range: tuple[int, int] = (1, 10),
        num_samples: int = 100,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.object_count_range = object_count_range
        self.object_size_range = object_size_range

    def __getitem__(self, idx: int) -> tuple[str, Tensor, Tensor, Tensor]:
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
        image, boxes, _, labels = poly()
        boxes, indices = filter_size(boxes, lambda x: x > 36)
        labels = labels[indices]
        return (
            "",
            image,
            boxes.float(),
            labels,
        )

    def __len__(self) -> int:
        return self.num_samples
