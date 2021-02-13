from typing import *
import torch
from torch.utils.data import Dataset
from object_detection.entities import (
    Image,
    ImageSize,
    Boxes,
    ImageId,
    Labels,
)
import random


class RandomDataset(Dataset):
    def __init__(self, image_size: ImageSize, num_samples: int = 8) -> None:
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, idx: int) -> Tuple[str, Image, Boxes, Labels]:
        image = torch.rand((3, *self.image_size), dtype=torch.float32)
        boxes = torch.rand((random.randint(1, 9), 4), dtype=torch.float32).clamp(
            0, max(self.image_size)
        )
        labels = torch.zeros((len(boxes),))
        return (
            "",
            Image(image),
            Boxes(boxes.float()),
            Labels(labels),
        )

    def __len__(self) -> int:
        return self.num_samples
