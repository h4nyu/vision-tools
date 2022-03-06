from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import torch
from albumentations.pytorch.transforms import ToTensorV2
from nanoid import generate as nanoid
from toolz.curried import (
    filter,
    frequencies,
    groupby,
    map,
    mapcat,
    pipe,
    sorted,
    topk,
    valmap,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.ops import box_area, box_convert
from typing_extensions import TypedDict

from coco_annotator import CocoCategory, CocoImage
from vision_tools.interface import Classification

from .metrics import MeanAveragePrecisionK

Annotation = TypedDict(
    "Annotation",
    {
        "image_file": str,
        "species": Optional[str],
        "individual_id": Optional[str],
        "label": Optional[int],
        "individual_samples": Optional[int],
    },
)

Submission = TypedDict(
    "Submission",
    {
        "image_file": str,
        "distances": List[float],
        "individual_ids": List[str],
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
    individual_samples = pipe(
        df.iterrows(),
        groupby(lambda x: x[1]["individual_id"]),
        valmap(len),
    )
    for _, csv_row in df.iterrows():
        rows.append(
            Annotation(
                image_file=os.path.basename(csv_row["image"]),
                species=csv_row["species"],
                individual_id=csv_row["individual_id"],
                label=label_map[csv_row["individual_id"]],
                individual_samples=individual_samples[csv_row["individual_id"]],
            )
        )
    rows = pipe(
        rows,
        map(correct_species),
        list,
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
    min_samples: int = 0,
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


def filter_in_annotations(
    annotations: list[Annotation],
    ref_annots: list[Annotation],
) -> list[Annotation]:
    individual_ids = pipe(
        ref_annots,
        map(lambda x: x["individual_id"]),
        set,
    )

    return pipe(
        annotations,
        filter(lambda x: x["individual_id"] in individual_ids),
        list,
    )


def create_croped_dataset(
    box_annotations: list[dict],
    source_dir: str,
    dist_dir: str,
    padding: float = 0.05,
    annotations: Optional[list[Annotation]] = None,
    suffix: str = ".jpg",
) -> list[Annotation]:
    image_boxes = pipe(
        box_annotations,
        groupby(lambda x: x["image_id"]),
    )

    annotation_map = pipe(
        annotations or [],
        map(lambda x: (Path(x["image_file"]).stem, x)),
        dict,
    )
    croped_annots: list[Annotation] = []
    for image_id in image_boxes.keys():
        boxes = image_boxes[image_id]
        (box,) = topk(1, boxes, key=lambda x: x["score"])
        source_path = os.path.join(source_dir, f"{image_id}{suffix}")
        im = PIL.Image.open(source_path)
        box_width = box["x2"] - box["x1"]
        box_height = box["y2"] - box["y1"]
        padding_x = box_width * padding
        padding_y = box_height * padding
        croped_box = [
            box["x1"] - padding_x,
            box["y1"] - padding_y,
            box["x2"] + padding_x,
            box["y2"] + padding_y,
        ]
        im_crop = im.crop(croped_box)
        image_annot = annotation_map.get(
            image_id,
            {
                "species": None,
                "individual_id": None,
                "label": None,
            },
        )
        dist_file_name = f"{image_id}-{nanoid(size=4)}{suffix}"
        dist_path = os.path.join(dist_dir, dist_file_name)
        im_crop.save(dist_path)
        croped_annots.append(
            Annotation(
                {
                    "image_file": dist_file_name,
                    "species": image_annot["species"],
                    "individual_id": image_annot["individual_id"],
                    "label": image_annot["label"],
                    "individual_samples": image_annot["individual_samples"],
                }
            )
        )
    return croped_annots


TrainTransform = lambda cfg: A.Compose(
    [
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=cfg["hue_p"]
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.9),
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_NEAREST,
        ),
        A.RandomResizedCrop(
            height=cfg["image_height"],
            width=cfg["image_width"],
            scale=(0.8, 1.0),
            interpolation=cv2.INTER_NEAREST,
            p=cfg["random_resized_crop_p"],
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


def collate_fn(
    batch: list[tuple[Classification, Annotation]]
) -> tuple[dict[str, Tensor], list[Annotation]]:
    images: list[Tensor] = []
    label_batch: list[Tensor] = []
    annots: list[Annotation] = []
    for row, annt in batch:
        images.append(row["image"])
        label_batch.append(row["label"])
        annots.append(annt)
    return (
        dict(
            image_batch=torch.stack(images),
            label_batch=torch.stack(label_batch),
        ),
        annots,
    )


def add_new_individual(
    submissions: list[Submission],
    threshold: float,
) -> list[Submission]:
    new_submissions = []
    for sub in submissions:
        has_added = False
        individua_ids = []
        new_sub = {
            **sub,
        }
        for (id, distance) in zip(sub["individual_ids"], sub["distances"]):
            if not has_added and distance < threshold:
                individua_ids.append("new_individual")
                has_added = True
            individua_ids.append(id)
        new_sub["individual_ids"] = individua_ids[: len(sub["individual_ids"])]
        new_submissions.append(new_sub)
    return new_submissions


def search_threshold(
    train_annotations: list[Annotation],
    val_annotations: list[Annotation],
    submissions: list[Submission],
    thresholds: list[float],
) -> float:
    train_individual_ids = pipe(
        train_annotations,
        map(lambda x: x["individual_id"]),
        set,
    )
    val_individual_ids = pipe(
        val_annotations,
        map(lambda x: x["individual_id"]),
        set,
    )
    new_individual_ids = val_individual_ids - train_individual_ids

    val_annotations = pipe(
        val_annotations,
        map(
            lambda x: {
                **x,
                "individual_id": "new_individual"
                if x["individual_id"] in new_individual_ids
                else x["individual_id"],
            }
        ),
        list,
    )
    val_annot_map = pipe(
        val_annotations,
        map(
            lambda x: (x["image_file"], x),
        ),
        dict,
    )
    all_individual_ids = train_individual_ids | set(["new_individual"])
    label_map = pipe(
        all_individual_ids,
        sorted,
        enumerate,
        map(lambda x: (x[1], x[0])),
        dict,
    )
    results = []

    for thr in thresholds:
        metric = MeanAveragePrecisionK()
        thr_submissions = add_new_individual(submissions, thr)
        for sub in thr_submissions:
            labels_at_k = (
                pipe(
                    sub["individual_ids"],
                    map(lambda x: label_map[x]),
                    list,
                    torch.tensor,
                )
                .long()
                .unsqueeze(0)
            )
            annot = val_annot_map[sub["image_file"]]
            labels = pipe(
                [annot["individual_id"]],
                map(lambda x: label_map[x]),
                list,
                torch.tensor,
            ).long()
            metric.update(labels_at_k, labels)
        score, _ = metric.value
        results.append(
            {
                "threshold": thr,
                "score": score,
            }
        )
    return results

    # print(label_map)

    # # ids = pipe(
    # #     annotations,
    # #     map(lambda x: x["individual_id"]),
    # #     set,
    # # )
