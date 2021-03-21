from typing import *
from typing_extensions import TypedDict

import torch, os, torchvision, PIL
import numpy as np
from torch.utils.data import Dataset
from vnet import Image, Boxes, Labels
from dataclasses import dataclass
import pandas as pd
from toolz.curried import pipe, partition, map, filter
from joblib import Memory
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from bench.kuzushiji import config
from vnet.transforms import normalize, inv_normalize
from vnet import Points
from sklearn.model_selection import StratifiedKFold

location = "/tmp"
memory = Memory(location, verbose=0)


Row = TypedDict(
    "Row",
    {
        "id": str,
        "image_path": str,
        "width": int,
        "height": int,
        "boxes": Boxes,
        "labels": Labels,
    },
)

SubRow = TypedDict(
    "SubRow",
    {
        "id": str,
        "points": Points,
        "labels": Labels,
    },
)


def save_submission(rows: List[SubRow], code_map: Dict[str, int], fpath: str) -> None:
    codes = list(code_map.keys())
    csv_rows: List[Tuple[str, str]] = []
    for row in rows:
        id = row["id"]
        points = row["points"]
        labels = row["labels"]
        out_str = ""
        for point, label in zip(points, labels):
            out_str += f"{codes[label]} {int(point[0])} {int(point[1])} "
        csv_rows.append((id, out_str))
    df = pd.DataFrame(csv_rows, columns=["image_id", "labels"])
    df.to_csv(fpath, index=False)


def read_code_map(fp: str) -> Dict[str, int]:
    df = pd.read_csv(fp)
    label_map: Dict[str, int] = Dict()
    for id, csv_row in df.iterrows():
        code, value = csv_row
        label_map[code] = id

    return label_map


@memory.cache
def read_train_rows(root_dir: str) -> List[Row]:
    row_path = os.path.join(root_dir, "train.csv")
    code_path = os.path.join(root_dir, "unicode_translation.csv")
    codes = read_code_map(code_path)
    df = pd.read_csv(row_path)
    rows: List[Row] = []
    for _, csv_row in df.iterrows():
        id = csv_row["image_id"]
        labels = []
        boxes = []
        for code, x0, y0, w, h in partition(5, csv_row["labels"].split(" ")):
            labels.append(codes[code])
            boxes.append(
                [
                    float(x0),
                    float(y0),
                    float(x0) + float(w),
                    float(y0) + float(h),
                ]
            )
        image_path = os.path.join(root_dir, "images", f"{id}.jpg")
        pil_img = PIL.Image.open(image_path)
        row: Row = Dict(
            id=id,
            image_path=image_path,
            width=pil_img.width,
            height=pil_img.height,
            boxes=Boxes(torch.tensor(boxes)),
            labels=Labels(torch.tensor(labels)),
        )
        rows.append(row)
    return rows


# @memory.cache
def read_test_rows(root_dir: str) -> List[Row]:
    row_path = os.path.join(root_dir, "sample_submission.csv")
    df = pd.read_csv(row_path)
    rows: List[Row] = []
    for _, csv_row in df.iterrows():
        id = csv_row["image_id"]
        image_path = os.path.join(root_dir, "images", f"{id}.jpg")
        pil_img = PIL.Image.open(image_path)
        row: Row = Dict(
            id=id,
            image_path=image_path,
            width=pil_img.width,
            height=pil_img.height,
            boxes=Boxes(torch.tensor([])),
            labels=Labels(torch.tensor([])),
        )
        rows.append(row)
    return rows


def kfold(
    rows: List[Row], n_splits: int, fold_idx: int = 0
) -> Tuple[List[Row], List[Row]]:
    skf = StratifiedKFold()
    x = range(len(rows))
    y = pipe(rows, map(lambda x: len(x["labels"])), List)
    pair_List: List[Tuple[List[int], List[int]]] = []
    for train_index, test_index in skf.split(x, y):
        pair_List.append((train_index, test_index))
    train_index, test_index = pair_List[fold_idx]
    return (
        [rows[i] for i in train_index],
        [rows[i] for i in test_index],
    )


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_visibility=0.75)

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size, min_width=config.image_size, border_mode=0
        ),
        A.ShiftScaleRotate(p=0.9, rotate_limit=10, scale_limit=0.2, border_mode=0),
        A.RandomCrop(config.image_size, config.image_size, p=1.0),
        A.ToGray(),
        normalize,
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)
default_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size, min_width=config.image_size, border_mode=0
        ),
        normalize,
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class KuzushijiDataset(Dataset):
    def __init__(
        self,
        rows: List[Row],
        transforms: Any = None,
    ) -> None:
        self.rows = rows
        self.transforms = default_transforms if transforms is None else transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[Image, Boxes, Labels, Image, Row]:
        row = self.rows[idx]
        id = row["id"]
        pil_img = PIL.Image.open(row["image_path"])
        img_arr = np.array(pil_img)
        original_img = Image(T.ToTensor()(img_arr))
        transformed = self.transforms(
            image=img_arr,
            bboxes=torchvision.ops.clip_boxes_to_image(
                row["boxes"], (pil_img.height, pil_img.width)
            ),
            labels=row["labels"],
        )
        img = Image(transformed["image"])
        boxes = Boxes(torch.tensor(transformed["bboxes"]))
        labels = Labels(torch.tensor(transformed["labels"]))
        return img, boxes, labels, original_img, row
