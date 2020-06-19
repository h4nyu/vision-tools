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
from app.entities import Boxes
from albumentations.pytorch.transforms import ToTensorV2
from skimage.io import imread


from pathlib import Path
from app import config
from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from .entities import Annotations, Annotation
from .models.utils import NestedTensor, box_xyxy_to_cxcywh

Row = t.Tuple[Tensor, Tensor, str]
Batch = t.Sequence[Row]


def collate_fn(batch: Batch) -> t.Tuple[Tensor, Annotations, t.List[str]]:
    images: t.List[Tensor] = []
    annots: Annotations = []
    ids: t.List[str] = []
    for img, boxes, id in batch:
        images.append(img)
        _, h, w = img.shape
        annots.append(Boxes(boxes=boxes, w=w, h=h, fmt="cxcywh",))
        ids.append(id)
    images_tensor = torch.stack(images)
    return images_tensor, annots, ids


train_transforms = albm.Compose(
    [
        albm.VerticalFlip(),
        albm.RandomRotate90(),
        albm.HorizontalFlip(),
        albm.RandomBrightness(limit=0.1),
    ],
    bbox_params=dict(format="albumentations", label_fields=["labels"]),
)


transforms = albm.Compose(
    [
        #  albm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def get_img(image_id: str) -> t.Any:
    image_path = f"{config.image_dir}/{image_id}.jpg"
    return (imread(image_path) / 255).astype(np.float32)


class WheatDataset(Dataset):
    def __init__(
        self,
        annotations: Annotations,
        mode: t.Literal["train", "test"] = "train",
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        self.rows = annotations
        self.mode = mode
        self.cache: t.Dict[str, t.Any] = dict()
        self.use_cache = use_cache
        self.image_dir = Path(config.image_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Any:
        row = self.rows[index]
        image = get_img(row.id)
        boxes = row.boxes
        labels = torch.zeros(len(boxes)).long()
        if self.mode == "train":
            res = train_transforms(image=image, bboxes=boxes, labels=labels)
            image = res["image"]
            boxes = torch.tensor(res["bboxes"], dtype=torch.float32)

        image = transforms(image=image)["image"].float()
        boxes = box_xyxy_to_cxcywh(boxes)
        return image, boxes, row.id


class PreditionDataset(Dataset):
    def __init__(self, csv_path: str = config.submition_csv,) -> None:
        self.rows = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, str]:
        row = self.rows.iloc[index]
        id = row["image_id"]
        image = get_img(id)
        image = transforms(image=image)["image"].float()
        return image, id
