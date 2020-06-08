import typing as t
import numpy as np
import pandas as pd
import torch
import torchvision
import PIL

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from .entities import Images
from .models.utils import NestedTensor

Target = t.TypedDict("Target", {"boxes": Tensor, "labels": Tensor})
Row = t.Tuple[Tensor, Target]
Batch = t.Sequence[Row]


def collate_fn(batch: Batch) -> t.Tuple[NestedTensor, t.List[Target]]:
    images: t.List[Tensor] = []
    targets: t.List[Target] = []
    for img, tgt in batch:
        targets.append(tgt)
        images.append(img)

    images_tensor = torch.stack(images)
    b, _, h, w = images_tensor.shape
    mask = torch.zeros((b, h, w), dtype=torch.bool)
    return NestedTensor(images_tensor, mask), targets


class WheatDataset(Dataset):
    def __init__(
        self, images: Images, mode: t.Literal["train", "test"] = "train"
    ) -> None:
        super().__init__()
        self.rows = list(images.values())
        self.mode = mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Any:
        row = self.rows[index]
        image = ToTensorV2()(image=row.get_arr())["image"]
        boxes = torch.tensor(
            [x.to_arr() for x in row.bboxes], dtype=torch.float32
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=row.width)
        boxes[:, 1::2].clamp_(min=0, max=row.height)
        labels = torch.ones(boxes.shape[:1]).long()
        target: Target = {
            "boxes": boxes,
            "labels": labels,
        }
        return image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, ann_file: str, img_folder: str, transforms: t.Any = None,
    ) -> None:
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Target]:
        img, annots = super().__getitem__(idx)
        w, h = img.size
        boxes = torch.tensor([x["bbox"] for x in annots], dtype=torch.float32).reshape(
            -1, 4
        )
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        labels = torch.tensor([x["category_id"] for x in annots], dtype=torch.int64,)

        target: Target = {"boxes": boxes, "labels": labels}
        return img, target
