import torch, os, torchvision, PIL
import numpy as np
from torch.utils.data import Dataset
from vnet import Image, Boxes, Labels
from dataclasses import dataclass
from typing import Any, TypedDict
import pandas as pd
import cytoolz as tlz
from joblib import Memory
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from bench.kuzushiji import config
from vnet.transforms import normalize, inv_normalize

location = "/tmp"
memory = Memory(location, verbose=0)


Row = TypedDict(
    "Row",
    {
        "id": str,
        "image_fname": str,
        "boxes": Boxes,
        "labels": Labels,
    },
)


def read_code_map(fp: str) -> dict[str, int]:
    df = pd.read_csv(fp)
    label_map: dict[str, int] = dict()
    for id, csv_row in df.iterrows():
        code, value = csv_row
        label_map[code] = id

    return label_map


@memory.cache
def read_rows(root_dir: str) -> list[Row]:
    row_path = os.path.join(root_dir, "train.csv")
    code_path = os.path.join(root_dir, "unicode_translation.csv")
    codes = read_code_map(code_path)
    df = pd.read_csv(row_path)
    rows: list[Row] = []
    for _, csv_row in df.iterrows():
        id = csv_row["image_id"]
        labels = []
        boxes = []
        for code, x0, y0, w, h in tlz.partition(5, csv_row["labels"].split(" ")):
            labels.append(codes[code])
            boxes.append(
                [
                    float(x0),
                    float(y0),
                    float(x0) + float(w),
                    float(y0) + float(h),
                ]
            )
        row: Row = dict(
            id=id,
            image_fname=f"{id}.jpg",
            boxes=Boxes(torch.tensor(boxes)),
            labels=Labels(torch.tensor(labels)),
        )
        rows.append(row)

    return rows


bbox_params = dict(format="pascal_voc", label_fields=["labels"])
default_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=config.image_width),
        normalize,
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class KuzushijiDataset(Dataset):
    def __init__(
        self,
        rows: list[Row],
        image_dir: str = config.image_dir,
        transforms: Any = None,
    ) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.transforms = transforms or default_transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, Image, Boxes, Labels]:
        row = self.rows[idx]
        id = row["id"]
        image_path = os.path.join(self.image_dir, row["image_fname"])
        pil_img = np.asarray(PIL.Image.open(image_path))
        transformed = self.transforms(
            image=pil_img, bboxes=row["boxes"], labels=row["labels"]
        )
        img = Image(transformed["image"])
        boxes = Boxes(transformed["bboxes"])
        print(boxes)
        labels = Labels(transformed["labels"])
        return id, img, boxes, labels
