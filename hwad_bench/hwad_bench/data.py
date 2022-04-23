from __future__ import annotations

import itertools
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
from sklearn.preprocessing import LabelEncoder
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
from tqdm import tqdm
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
        "image_id": str,
        "distances": List[float],
        "individual_ids": List[str],
    },
)

SpecieLabels = pipe(
    [
        "beluga_whale",
        "blue_whale",
        "brydes_whale",
        "cuviers_beaked_whale",
        "false_killer_whale",
        "fin_whale",
        "gray_whale",
        "humpback_whale",
        "killer_whale",
        "long_finned_pilot_whale",
        "melon_headed_whale",
        "minke_whale",
        "pygmy_killer_whale",
        "sei_whale",
        "short_finned_pilot_whale",
        "southern_right_whale",
        "bottlenose_dolphin",
        "commersons_dolphin",
        "common_dolphin",
        "dusky_dolphin",
        "frasiers_dolphin",
        "pantropic_spotted_dolphin",
        "rough_toothed_dolphin",
        "spinner_dolphin",
        "spotted_dolphin",
        "white_sided_dolphin",
    ],
    sorted,
    enumerate,
    map(lambda x: (x[1], x[0])),
    dict,
)


def correct_species(
    annotation: Annotation,
) -> Annotation:
    species = annotation["species"]
    if species == "bottlenose_dolpin":
        annotation["species"] = "bottlenose_dolphin"
    elif species == "kiler_whale":
        annotation["species"] = "killer_whale"
    elif species == "beluga":
        annotation["species"] = "beluga_whale"
    elif species == "globis":
        annotation["species"] = "short_finned_pilot_whale"
    elif species == "pilot_whale":
        annotation["species"] = "short_finned_pilot_whale"

    return annotation


def summary(
    rows: list[Annotation],
) -> dict:
    all_species = pipe(rows, map(lambda x: x["species"]), set)
    individual_id_count = pipe(rows, map(lambda x: x["individual_id"]), set, len)
    class_freq = pipe(
        rows,
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
        "count": len(rows),
        "species_count": len(all_species),
        "individual_id_count": individual_id_count,
        "all_species": all_species,
        "class_freq": class_freq,
    }


def merge_to_coco_rows(rows: list[Annotation]) -> dict[str, list]:
    coco_categories = pipe(
        rows,
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
    coco_rows = []
    coco_images: list[dict] = []
    for i, annt in enumerate(rows):
        coco_images.append(
            {
                "id": i,
                "file_name": annt["image_file"],
            }
        )
        coco_rows.append(
            {
                "id": i,
                "image_id": i,
                "category_id": categories_id_map[annt["species"]],
            }
        )

    return {
        "images": coco_images,
        "categories": coco_categories,
        "rows": coco_rows,
    }


def read_rows(file_path: str) -> list:
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


def filter_rows_by_fold(
    rows: list[Annotation],
    fold: list[dict],
) -> list[Annotation]:
    image_ids = pipe(
        fold,
        map(lambda x: Path(x["image"]).stem),
        set,
    )
    filtered_rows = []
    for annt in rows:
        image_id = Path(annt["image_file"]).stem.split(".")[0]
        if image_id in image_ids:
            filtered_rows.append(annt)
    return filtered_rows


def merge_rows_by_image_id(
    *args: list[Annotation],
) -> list[Annotation]:
    rows_set = {}
    for annt in itertools.chain(*args):
        image_id = annt["image_file"].split(".")[0]
        rows_set[image_id] = annt
    return list(rows_set.values())


def filter_in_rows(
    rows: list[Annotation],
    ref_annots: list[Annotation],
) -> list[Annotation]:
    individual_ids = pipe(
        ref_annots,
        map(lambda x: x["individual_id"]),
        set,
    )

    return pipe(
        rows,
        filter(lambda x: x["individual_id"] in individual_ids),
        list,
    )


def create_croped_dataset(
    box_rows: list[dict],
    source_dir: str,
    dist_dir: str,
    padding: float = 0.125,
    rows: Optional[list[Annotation]] = None,
    part: str = "body",
    suffix: str = ".jpg",
) -> list[Annotation]:
    image_boxes = pipe(
        box_rows,
        groupby(lambda x: x.get("image_id", None) or x["image"].split(".")[0]),
    )

    annotation_map = pipe(
        rows or [],
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
                "individual_samples": None,
            },
        )
        dist_file_name = f"{image_id}.{part}{suffix}"
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
    print(croped_annots)
    return croped_annots


TrainTransform = lambda cfg: A.Compose(
    [
        A.Affine(
            rotate=(cfg["rot0"], cfg["rot1"]),
            shear=(cfg["shear0"], cfg["shear1"]),
            translate_percent=(cfg["trans0"], cfg["trans1"]),
            interpolation=cv2.INTER_CUBIC,
            p=cfg["affine_p"],
        ),
        A.RandomResizedCrop(
            height=cfg["image_height"],
            width=cfg["image_width"],
            scale=(0.9, 1.0),
            ratio=(0.75, 1.3333333333),
        ),
        A.ToGray(p=cfg["to_gray_p"]),
        A.GaussianBlur(
            blur_limit=(cfg["blur_limit0"], cfg["blur_limit1"]),
            p=cfg["gaussian_blur_p"],
        ),
        A.GaussNoise(p=cfg["gaussian_noise_p"]),
        A.RandomGridShuffle(
            grid=(cfg["grid_0"], cfg["grid_1"]), p=cfg["grid_shuffle_p"]
        ),
        A.Posterize(p=cfg["posterize_p"]),
        A.RandomBrightnessContrast(p=cfg["brightness_contrast_p"]),
        A.Cutout(p=cfg["cutout_p"]),
        A.RandomSnow(p=cfg["snow_p"]),
        A.RandomRain(p=cfg["rain_p"]),
        ToTensorV2(),
    ],
)

Transform = lambda cfg: A.Compose(
    [
        A.Resize(
            height=cfg["image_height"],
            width=cfg["image_width"],
            interpolation=cv2.INTER_CUBIC,
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
        self.id_map = pipe(
            rows,
            map(lambda x: (x["label"], x["individual_id"])),
            dict,
        )
        self.label_map = pipe(
            rows,
            map(lambda x: (x["individual_id"], x["label"])),
            dict,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], Annotation]:
        row = self.rows[idx]
        image_path = os.path.join(self.image_dir, row["image_file"])
        im = PIL.Image.open(image_path)
        if im.mode == "L":
            im = im.convert("RGB")
        img_arr = np.array(im)
        transformed = self.transform(
            image=img_arr,
        )
        image = (transformed["image"] / 255).float()
        label = torch.tensor(row["label"] or 0)
        species = row["species"]
        suplabel = torch.tensor(SpecieLabels[species] if species is not None else 0)
        sample = dict(
            image=image,
            label=label,
            suplabel=suplabel,
        )
        return sample, row


def collate_fn(
    batch: list[tuple[dict[str, Tensor], Annotation]]
) -> tuple[dict[str, Tensor], list[Annotation]]:
    images: list[Tensor] = []
    label_batch: list[Tensor] = []
    suplabel_batch: list[Tensor] = []
    annots: list[Annotation] = []
    for row, annt in batch:
        images.append(row["image"])
        label_batch.append(row["label"])
        suplabel_batch.append(row["suplabel"])
        annots.append(annt)
    return (
        dict(
            image_batch=torch.stack(images),
            label_batch=torch.stack(label_batch),
            suplabel_batch=torch.stack(suplabel_batch),
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
        new_sub: Any = {
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
