import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torchvision.transforms as T
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import GroupKFold
from toolz.curried import count, filter, map, partition, pipe
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.ops import box_convert, clip_boxes_to_image
from typing_extensions import TypedDict

from vision_tools.interface import Detection, TrainBatch
from vision_tools.transforms import RandomCutAndPaste, inv_normalize, normalize

Row = TypedDict(
    "Row",
    {
        "image_id": str,
        "video_id": int,
        "video_frame": int,
        "sequence": int,
        "image_path": str,
        "boxes": Tensor,
    },
)


def read_train_rows(dataset_dir: str) -> List[Row]:
    row_path = os.path.join(dataset_dir, "train.corrected.csv")
    df = pd.read_csv(row_path)
    rows: List[Row] = []
    subsequence = -1
    prev_count = -1

    for _, csv_row in df.iterrows():
        annotations = json.loads(csv_row["annotations"].replace("'", '"'))
        image_id = csv_row["image_id"]
        video_id = csv_row["video_id"]
        sequence = csv_row["sequence"]
        video_frame = csv_row["video_frame"]
        image_path = os.path.join(
            dataset_dir, "train_images", f"video_{video_id}/{video_frame}.jpg"
        )
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
                image_path=image_path,
                boxes=boxes,
                sequence=sequence,
            )
        )

    return rows


def kfold(
    rows: List[Row], n_splits: int, fold_idx: int = 0
) -> Tuple[List[Row], List[Row]]:
    skf = GroupKFold(n_splits=n_splits)
    x = range(len(rows))
    y = pipe(rows, map(lambda x: x["video_id"]), list)
    groups = pipe(rows, map(lambda x: x["sequence"]), list)
    pair_list: List[Tuple[List[int], List[int]]] = []
    for train_index, test_index in skf.split(x, y, groups=groups):
        pair_list.append((train_index, test_index))
    train_index, test_index = pair_list[fold_idx]
    return (
        [rows[i] for i in train_index],
        [rows[i] for i in test_index],
    )


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_area=14)

TrainTransform = lambda cfg: A.Compose(
    [
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=cfg["hue_p"]
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=cfg["vflip_p"]),
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_NEAREST,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)
Transform = lambda cfg: A.Compose(
    [
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=PIL.Image.BILINEAR,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

InferenceTransform = lambda cfg: A.Compose(
    [
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=PIL.Image.BILINEAR,
        ),
        ToTensorV2(),
    ],
)


class COTSDataset(Dataset):
    def __init__(
        self,
        rows: List[Row],
        transform: Any,
        random_cut_and_paste: Optional[RandomCutAndPaste] = None,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.random_cut_and_paste = random_cut_and_paste

    def __str__(self) -> str:
        string = ""
        string += "\tlen = %d\n" % len(self)
        zero_count = pipe(self.rows, filter(lambda x: x["boxes"].shape[0] == 0), count)
        non_zero_count = pipe(
            self.rows, filter(lambda x: x["boxes"].shape[0] != 0), count
        )
        string += "\tzero_count     = %5d (%0.2f)\n" % (
            zero_count,
            zero_count / len(self.rows),
        )
        string += "\tnon_zero_count = %5d (%0.2f)\n" % (
            non_zero_count,
            non_zero_count / len(self.rows),
        )
        return string

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[Detection, Row]:
        row = self.rows[idx]
        id = row["image_id"]
        video_id = row["video_id"]
        video_frame = row["video_frame"]
        image_path = row["image_path"]
        img_arr = np.array(PIL.Image.open(image_path))
        labels = torch.zeros(len(row["boxes"]))
        transformed = self.transform(
            image=img_arr,
            bboxes=clip_boxes_to_image(row["boxes"], img_arr.shape[:2]),
            labels=labels,
        )
        image = (transformed["image"] / 255).float()
        boxes = torch.tensor(transformed["bboxes"]).float()
        labels = torch.zeros(len(boxes)).long()
        confs = torch.ones(len(labels)).float()
        sample = Detection(
            image=image,
            boxes=boxes,
            labels=labels,
            confs=confs,
        )
        if self.random_cut_and_paste is not None:
            sample = self.random_cut_and_paste(sample)
        return sample, row


def to_submission_string(boxes: Tensor, confs: Tensor) -> str:
    cxcywhs = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
    out_str = ""
    for i, (cxcywh, conf) in enumerate(zip(cxcywhs, confs)):
        out_str += f"{conf:.4f} {cxcywh[0]:.4f} {cxcywh[1]:.4f} {cxcywh[2]:.4f} {cxcywh[3]:.4f} "
    return out_str.strip(" ")


def filter_empty_boxes(rows: List[Row]) -> List[Row]:
    return pipe(
        rows,
        filter(lambda x: len(x["boxes"]) > 0),
        list,
    )


def keep_ratio(rows: List[Row]) -> List[Row]:
    """keep zero and non-zero boxes with 1:4"""
    no_zero_rows = pipe(rows, filter_empty_boxes, list)
    non_zero_count = len(no_zero_rows)
    zero_rows = pipe(rows, filter(lambda x: len(x["boxes"]) == 0), list)[
        : int(non_zero_count) * 4
    ]
    return no_zero_rows + zero_rows


def collate_fn(
    batch: List[Tuple[Detection, Row]],
) -> TrainBatch:
    images: List[Tensor] = []
    box_batch: List[Tensor] = []
    label_batch: List[Tensor] = []
    conf_batch: List[Tensor] = []
    for row, _ in batch:
        images.append(row["image"])
        box_batch.append(row["boxes"])
        label_batch.append(row["labels"])
        conf_batch.append(row["confs"])
    return TrainBatch(
        image_batch=torch.stack(images),
        box_batch=box_batch,
        label_batch=label_batch,
        conf_batch=conf_batch,
    )
