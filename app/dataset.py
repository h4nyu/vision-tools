import typing as t
import numpy as np
import pandas as pd
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
import albumentations as albm
from albumentations.pytorch.transforms import ToTensorV2


from pathlib import Path
from app import config
from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from .entities import Annotations, Annotation
from .models.utils import NestedTensor, box_xyxy_to_cxcywh

Target = t.TypedDict("Target", {"boxes": Tensor, "labels": Tensor})
Row = t.Tuple[Tensor, Target]
Batch = t.Sequence[Row]
Targets = t.List[Target]


def collate_fn(batch: Batch) -> t.Tuple[Tensor, t.List[Target]]:
    images: t.List[Tensor] = []
    targets: t.List[Target] = []
    for img, tgt in batch:
        targets.append(tgt)
        images.append(img)
    images_tensor = torch.stack(images)
    return images_tensor, targets


train_transforms = albm.Compose(
    [
        albm.VerticalFlip(),
        albm.RandomRotate90(),
        albm.HorizontalFlip(),
        albm.RandomBrightness(limit=0.1),
        albm.RandomShadow(p=0.5),
    ],
    bbox_params=dict(format="albumentations", label_fields=["labels"]),
)


transforms = albm.Compose(
    [
        #  albm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class WheatDataset(Dataset):
    def __init__(
        self,
        annotations: Annotations,
        mode: t.Literal["train", "test"] = "train",
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        self.rows = list(annotations.values())
        self.mode = mode
        self.cache: t.Dict[str, t.Any] = dict()
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.rows)

    def get_img(self, row: Annotation) -> t.Any:
        if row.id not in self.cache:
            self.cache[row.id] = row.get_img()
        return self.cache[row.id]

    def __getitem__(self, index: int) -> t.Any:
        row = self.rows[index]
        image = self.get_img(row) if self.use_cache else row.get_img()
        boxes = row.boxes
        labels = torch.zeros(len(boxes)).long()
        if self.mode == "train":
            res = train_transforms(image=image, bboxes=boxes, labels=labels)
            image = res["image"]
            boxes = torch.tensor(res["bboxes"], dtype=torch.float32)
            labels = res["labels"]

        image = transforms(image=image)["image"].float()
        boxes = box_xyxy_to_cxcywh(boxes)
        labels = torch.tensor(labels)
        target: Target = {
            "boxes": boxes,
            "labels": labels,
        }
        return image, target
