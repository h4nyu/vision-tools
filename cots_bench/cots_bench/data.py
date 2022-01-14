from typing import Any, TypedDict
from torch import Tensor
import json
import torch, os, torchvision, PIL
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
import pandas as pd
from toolz.curried import pipe, partition, map, filter
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from vision_tools.transforms import normalize, inv_normalize
from vision_tools.interface import TrainSample
from sklearn.model_selection import StratifiedKFold
from vision_tools.interface import TrainBatch, TrainSample
from torchvision.ops import box_convert, clip_boxes_to_image

location = "/tmp"

Row = TypedDict(
    "Row",
    {
        "image_id": str,
        "video_id": int,
        "video_frame": int,
        "boxes": Tensor,
    },
)

# SubRow = TypedDict(
#     "SubRow",
#     {
#         "id": str,
#         "points": Tensor,
#         "labels": Tensor,
#     },
# )


# def save_submission(rows: list[SubRow], code_map: dict[str, int], fpath: str) -> None:
#     codes = list(code_map.keys())
#     csv_rows: list[tuple[str, str]] = []
#     for row in rows:
#         id = row["id"]
#         points = row["points"]
#         labels = row["labels"]
#         out_str = ""
#         for point, label in zip(points, labels):
#             out_str += f"{codes[label]} {int(point[0])} {int(point[1])} "
#         csv_rows.append((id, out_str))
#     df = pd.DataFrame(csv_rows, columns=["image_id", "labels"])
#     df.to_csv(fpath, index=False)


def read_train_rows(root_dir: str) -> list[Row]:
    row_path = os.path.join(root_dir, "train.csv")
    df = pd.read_csv(row_path)
    rows: list[Row] = []
    for _, csv_row in df.iterrows():
        annotations = json.loads(csv_row["annotations"].replace("'", '"'))
        image_id = csv_row["image_id"]
        video_id = csv_row["video_id"]
        video_frame = csv_row["video_frame"]
        boxes = torch.zeros(0, 4)
        if len(annotations) > 0:
            boxes = box_convert(
                torch.tensor(
                    [[x["x"], x["y"], x["width"], x["height"]] for x in annotations]
                ),
                in_fmt="xywh",
                out_fmt="xyxy",
            )

        rows.append(
            Row(
                image_id=image_id,
                video_frame=video_frame,
                video_id=video_id,
                boxes=boxes,
            )
        )

    return rows


# def read_test_rows(root_dir: str) -> list[Row]:
#     row_path = os.path.join(root_dir, "sample_submission.csv")
#     df = pd.read_csv(row_path)
#     rows: list[Row] = []
#     for _, csv_row in df.iterrows():
#         id = csv_row["image_id"]
#         image_path = os.path.join(root_dir, f"{id}.jpg")
#         pil_img = PIL.Image.open(image_path)
#         row: Row = dict(
#             id=id,
#             image_path=image_path,
#             width=pil_img.width,
#             height=pil_img.height,
#             boxes=torch.tensor([]),
#             labels=torch.tensor([]),
#         )
#         rows.append(row)
#     return rows


# def kfold(
#     rows: list[Row], n_splits: int, fold_idx: int = 0
# ) -> tuple[list[Row], list[Row]]:
#     skf = StratifiedKFold()
#     x = range(len(rows))
#     y = pipe(rows, map(lambda x: len(x["labels"])), list)
#     pair_list: list[tuple[list[int], list[int]]] = []
#     for train_index, test_index in skf.split(x, y):
#         pair_list.append((train_index, test_index))
#     train_index, test_index = pair_list[fold_idx]
#     return (
#         [rows[i] for i in train_index],
#         [rows[i] for i in test_index],
#     )


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_visibility=0.75)

TrainTransform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.ColorJitter(p=0.7, brightness=0.2, contrast=0.3, hue=0.1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.ShiftScaleRotate(p=0.9, rotate_limit=10, scale_limit=0.2, border_mode=0),
        A.RandomCrop(image_size, image_size, p=1.0),
        A.ToGray(p=0.05),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)
Transform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class COTSDataset(Dataset):
    def __init__(
        self,
        rows: list[Row],
        transform: Any,
        image_dir: str,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TrainSample:
        row = self.rows[idx]
        id = row["image_id"]
        video_id = row["video_id"]
        video_frame = row["video_frame"]
        img_arr = np.array(PIL.Image.open(f"{self.image_dir}/video_{video_id}/{video_frame}.jpg"))
        labels = torch.zeros(len(row['boxes']))
        transformed = self.transform(
            image=img_arr,
            bboxes=row["boxes"],
            labels=labels,
        )
        image = (transformed["image"] / 255).float()
        boxes = torch.tensor(transformed["bboxes"]).float()
        labels = torch.zeros(len(boxes)).long()
        return TrainSample(
            id=id,
            image=image,
            boxes=boxes,
            labels=labels,
        )


def collate_fn(
    batch: list[TrainSample],
) -> TrainBatch:
    images: list[Tensor] = []
    box_batch: list[Tensor] = []
    label_batch: list[Tensor] = []
    for row in batch:
        images.append(row["image"])
        box_batch.append(row["boxes"])
        label_batch.append(row["labels"])
    return TrainBatch(
        image_batch=torch.stack(images),
        box_batch=box_batch,
        label_batch=label_batch,
    )
