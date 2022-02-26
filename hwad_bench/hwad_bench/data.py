from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import torch
from albumentations.pytorch.transforms import ToTensorV2
from nanoid import generate as nanoid
from toolz.curried import filter, frequencies, groupby, map, pipe, sorted, valmap
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from coco_annotator import CocoCategory, CocoImage
from vision_tools.interface import Classification

Annotation = TypedDict(
    "Annotation",
    {
        "image_file": str,
        "species": str,
        "individual_id": str,
        "label": int,
    },
)


def correct_species(
    annotation: Annotation,
) -> Annotation:
    species = annotation["species"]
    if species == "bottlenose_dolpin":
        annotation["species"] = "bottlenose_dolphin"
    elif species == "kiler_whale":
        annotation["species"] = "killer_whale"
    return annotation


def cleansing(
    annotations: list[Annotation],
) -> list[Annotation]:
    return pipe(
        annotations,
        map(correct_species),
        list,
    )


def summary(
    annotations: list[Annotation],
) -> dict:
    all_species = pipe(annotations, map(lambda x: x["species"]), set)
    individual_id_count = pipe(annotations, map(lambda x: x["individual_id"]), set, len)
    class_freq = pipe(
        annotations,
        groupby(lambda x: x["individual_id"]),
        valmap(len),
        lambda x: x.values(),
        frequencies,
        lambda x: x.items(),
        lambda x: sorted(x, key=lambda y: y[0]),
        map(lambda x: {"class_size": x[0], "count": x[1]}),
        list,
    )
    return {
        "count": len(annotations),
        "species_count": len(all_species),
        "individual_id_count": individual_id_count,
        "all_species": all_species,
        "class_freq": class_freq,
    }


def merge_to_coco_annotations(annotations: list[Annotation]) -> dict[str, list]:
    coco_categories = pipe(
        annotations,
        map(lambda x: x["species"]),
        set,
        enumerate,
        map(
            lambda x: {
                "id": x[0],
                "name": x[1],
                "supercategory": "animal",
            }
        ),
        list,
    )
    categories_id_map = pipe(
        coco_categories,
        map(lambda x: (x["name"], x["id"])),
        dict,
    )
    coco_annotations = []
    coco_images: list[dict] = []
    for i, annt in enumerate(annotations):
        coco_images.append(
            {
                "id": i,
                "file_name": annt["image_file"],
            }
        )
        coco_annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": categories_id_map[annt["species"]],
            }
        )

    return {
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_annotations,
    }


def read_annotations(file_path: str) -> list:
    df = pd.read_csv(file_path)
    rows: list[Annotation] = []
    label_map = pipe(
        df.iterrows(),
        map(lambda x: x[1]["individual_id"]),
        set,
        sorted,
        enumerate,
        map(lambda x: (x[1], x[0])),
        dict,
    )
    for _, csv_row in df.iterrows():
        rows.append(
            Annotation(
                image_file=os.path.basename(csv_row["image"]),
                species=csv_row["species"],
                individual_id=csv_row["individual_id"],
                label=label_map[csv_row["individual_id"]],
            )
        )
    return rows


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def read_csv(path: str) -> dict:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def filter_annotations_by_fold(
    annotations: list[Annotation],
    fold: list[dict],
    min_samples: int = 10,
) -> list[Annotation]:
    fold = pipe(
        fold,
        filter(lambda x: x["individual_samples"] >= min_samples),
        list,
    )
    image_ids = pipe(
        fold,
        map(lambda x: Path(x["image"]).stem),
        set,
    )
    filtered_annotations = []
    for annt in annotations:
        image_id = Path(annt["image_file"]).stem.split("-")[0]
        if image_id in image_ids:
            filtered_annotations.append(annt)
    return filtered_annotations


def create_croped_dataset(
    coco: dict,
    annotations: list[Annotation],
    source_dir: str,
    dist_dir: str,
    padding: float = 0.05,
) -> list[Annotation]:
    image_map = pipe(
        coco["images"],
        map(lambda x: (x["id"], x)),
        dict,
    )

    # filter multiple annotations
    coco_annotations = pipe(
        coco["annotations"],
        groupby(lambda x: x["image_id"]),
        lambda x: x.values(),
        filter(lambda x: len(x) == 1),
        map(lambda x: x[0]),
        list,
    )
    annotation_map = pipe(
        annotations,
        map(lambda x: (x["image_file"], x)),
        dict,
    )
    croped_annots: list[Annotation] = []
    for annot in coco_annotations:
        image = image_map[annot["image_id"]]
        image_annot = annotation_map[image["file_name"]]
        source_path = os.path.join(source_dir, image["file_name"])
        im = PIL.Image.open(source_path)
        coco_bbox = annot["bbox"]
        padding_x = coco_bbox[2] * padding
        padding_y = coco_bbox[3] * padding
        bbox = [
            coco_bbox[0] - padding_x,
            coco_bbox[1] - padding_y,
            coco_bbox[0] + coco_bbox[2] + padding_x,
            coco_bbox[1] + coco_bbox[3] + padding_y,
        ]
        im_crop = im.crop(bbox)
        stem = Path(image_annot["image_file"]).stem
        suffix = Path(image_annot["image_file"]).suffix
        dist_file_name = f"{stem}-{nanoid(size=4)}{suffix}"
        dist_path = os.path.join(dist_dir, dist_file_name)
        im_crop.save(dist_path)
        croped_annots.append(
            Annotation(
                {
                    "image_file": dist_file_name,
                    "species": image_annot["species"],
                    "individual_id": image_annot["individual_id"],
                    "label": image_annot["label"],
                }
            )
        )
    return croped_annots


TrainTransform = lambda cfg: A.Compose(
    [
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=cfg["hue_p"]
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.9),
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_NEAREST,
        ),
        ToTensorV2(),
    ],
)

Transform = lambda cfg: A.Compose(
    [
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_NEAREST,
        ),
        ToTensorV2(),
    ],
)


class HwadCropedDataset(Dataset):
    def __init__(
        self,
        rows: list[Annotation],
        transform: Callable,
        image_dir: str,
    ) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Classification, Annotation]:
        row = self.rows[idx]
        image_path = os.path.join(self.image_dir, row["image_file"])
        im = PIL.Image.open(image_path)
        if im.mode == "L":
            im = im.convert("RGB")
        img_arr = np.array(im)
        transformed = self.transform(
            image=img_arr,
            label=row["label"],
        )
        image = (transformed["image"] / 255).float()
        label = torch.tensor(transformed["label"])
        sample = Classification(
            image=image,
            label=label,
        )
        return sample, row


def collate_fn(batch: list[tuple[Classification, Annotation]]) -> dict[str, Tensor]:
    images: list[Tensor] = []
    label_batch: list[Tensor] = []
    for row, _ in batch:
        images.append(row["image"])
        label_batch.append(row["label"])
    return dict(
        image_batch=torch.stack(images),
        label_batch=torch.stack(label_batch),
    )


# @torch.no_grad()
# def accuracy(output:Tensor, target:Tensor, topk:tuple[int, int]=(1, 5)) -> Tensor:
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
