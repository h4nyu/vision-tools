from typing import Any, TypedDict
from torch import Tensor

import torch, os, torchvision, PIL
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
import pandas as pd
from toolz.curried import pipe, partition, map, filter
from joblib import Memory
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from vision_tools.transforms import normalize, inv_normalize
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
        "boxes": Tensor,
        "labels": Tensor,
    },
)

SubRow = TypedDict(
    "SubRow",
    {
        "id": str,
        "points": Tensor,
        "labels": Tensor,
    },
)


def save_submission(rows: list[SubRow], code_map: dict[str, int], fpath: str) -> None:
    codes = list(code_map.keys())
    csv_rows: list[tuple[str, str]] = []
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


def read_code_map(fp: str) -> dict[str, int]:
    df = pd.read_csv(fp)
    label_map: dict[str, int] = dict()
    for id, csv_row in df.iterrows():
        code, value = csv_row
        label_map[code] = id

    return label_map


@memory.cache
def read_train_rows(root_dir: str) -> list[Row]:
    row_path = os.path.join(root_dir, "train.csv")
    code_path = os.path.join(root_dir, "unicode_translation.csv")
    codes = read_code_map(code_path)
    df = pd.read_csv(row_path)
    rows: list[Row] = []
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
        image_path = os.path.join(root_dir, f"{id}.jpg")
        pil_img = PIL.Image.open(image_path)
        row: Row = dict(
            id=id,
            image_path=image_path,
            width=pil_img.width,
            height=pil_img.height,
            boxes=torch.tensor(boxes),
            labels=torch.tensor(labels),
        )
        rows.append(row)
    return rows


# @memory.cache
def read_test_rows(root_dir: str) -> list[Row]:
    row_path = os.path.join(root_dir, "sample_submission.csv")
    df = pd.read_csv(row_path)
    rows: list[Row] = []
    for _, csv_row in df.iterrows():
        id = csv_row["image_id"]
        image_path = os.path.join(root_dir, f"{id}.jpg")
        pil_img = PIL.Image.open(image_path)
        row: Row = dict(
            id=id,
            image_path=image_path,
            width=pil_img.width,
            height=pil_img.height,
            boxes=torch.tensor([]),
            labels=torch.tensor([]),
        )
        rows.append(row)
    return rows


def kfold(
    rows: list[Row], n_splits: int, fold_idx: int = 0
) -> tuple[list[Row], list[Row]]:
    skf = StratifiedKFold()
    x = range(len(rows))
    y = pipe(rows, map(lambda x: len(x["labels"])), list)
    pair_list: list[tuple[list[int], list[int]]] = []
    for train_index, test_index in skf.split(x, y):
        pair_list.append((train_index, test_index))
    train_index, test_index = pair_list[fold_idx]
    return (
        [rows[i] for i in train_index],
        [rows[i] for i in test_index],
    )


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_visibility=0.75)

# train_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=config.image_size),
#         A.PadIfNeeded(
#             min_height=config.image_size, min_width=config.image_size, border_mode=0
#         ),
#         A.ShiftScaleRotate(p=0.9, rotate_limit=10, scale_limit=0.2, border_mode=0),
#         A.RandomCrop(config.image_size, config.image_size, p=1.0),
#         A.ToGray(),
#         normalize,
#         ToTensorV2(),
#     ],
#     bbox_params=bbox_params,
# )
Transfrom = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=0
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class KuzushijiDataset(Dataset):
    def __init__(
        self,
        rows: list[Row],
        transforms: Any,
    ) -> None:
        self.rows = rows
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Row]:
        row = self.rows[idx]
        id = row["id"]
        pil_img = PIL.Image.open(row["image_path"])
        img_arr = np.array(pil_img)
        original_img = T.ToTensor()(img_arr)
        transformed = self.transforms(
            image=img_arr,
            bboxes=torchvision.ops.clip_boxes_to_image(
                row["boxes"], (pil_img.height, pil_img.width)
            ),
            labels=row["labels"],
        )
        img = transformed["image"] / 255
        boxes = torch.tensor(transformed["bboxes"])
        labels = torch.tensor(transformed["labels"])
        return img, boxes, labels, original_img, row
