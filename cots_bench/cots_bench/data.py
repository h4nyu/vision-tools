from typing import Any, List, Tuple
from typing_extensions import TypedDict
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
from sklearn.model_selection import GroupKFold
from vision_tools.interface import TrainBatch, TrainSample
from torchvision.ops import box_convert, clip_boxes_to_image


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


def read_train_rows(dataset_dir: str, skip_empty: bool = False) -> List[Row]:
    row_path = os.path.join(dataset_dir, "train.csv")
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
        else:
            if skip_empty:
                continue

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
    skf = GroupKFold()
    x = range(len(rows))
    y = pipe(rows, map(lambda x: len(x["boxes"])), list)
    groups = pipe(rows, map(lambda x: x["sequence"]), list)
    pair_list: List[Tuple[List[int], List[int]]] = []
    for train_index, test_index in skf.split(x, y, groups=groups):
        pair_list.append((train_index, test_index))
    train_index, test_index = pair_list[fold_idx]
    return (
        [rows[i] for i in train_index],
        [rows[i] for i in test_index],
    )


bbox_params = dict(format="pascal_voc", label_fields=["labels"], min_visibility=0.75)

TrainTransform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.ColorJitter(p=0.7, brightness=0.2, contrast=0.3, hue=0.1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.ShiftScaleRotate(p=0.9, rotate_limit=10, scale_limit=0.2, border_mode=0),
        A.RandomCrop(image_size, image_size, p=1.0),
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

InferenceTransform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        ToTensorV2(),
    ],
)


class COTSDataset(Dataset):
    def __init__(
        self,
        rows: List[Row],
        transform: Any,
    ) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TrainSample:
        row = self.rows[idx]
        id = row["image_id"]
        video_id = row["video_id"]
        video_frame = row["video_frame"]
        image_path = row["image_path"]
        img_arr = np.array(PIL.Image.open(image_path))
        labels = torch.zeros(len(row["boxes"]))
        confs = torch.ones(len(row["boxes"])).float()
        transformed = self.transform(
            image=img_arr,
            bboxes=clip_boxes_to_image(row["boxes"], img_arr.shape[:2]),
            labels=labels,
        )
        image = (transformed["image"] / 255).float()
        boxes = torch.tensor(transformed["bboxes"]).float()
        labels = torch.zeros(len(boxes)).long()
        return TrainSample(
            image=image,
            boxes=boxes,
            labels=labels,
            confs=confs,
        )


def to_submission_string(boxes: Tensor, confs: Tensor) -> str:
    cxcywhs = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
    out_str = ""
    for i, (cxcywh, conf) in enumerate(zip(cxcywhs, confs)):
        out_str += f"{conf:.4f} {cxcywh[0]:.4f} {cxcywh[1]:.4f} {cxcywh[2]:.4f} {cxcywh[3]:.4f} "
    return out_str.strip(" ")


def collate_fn(
    batch: List[TrainSample],
) -> TrainBatch:
    images: List[Tensor] = []
    box_batch: List[Tensor] = []
    label_batch: List[Tensor] = []
    conf_batch: List[Tensor] = []
    for row in batch:
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
