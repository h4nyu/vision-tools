from __future__ import annotations as _

from typing import Any

import pytest

from hwad_bench.data import (
    Annotation,
    HwadCropedDataset,
    TrainTransform,
    create_croped_dataset,
    filter_annotations_by_fold,
    merge_to_coco_annotations,
    read_annotations,
    read_csv,
)
from vision_tools.utils import load_config

dataset_cfg = load_config("config/dataset.yaml")


@pytest.fixture
def annotations() -> list[Annotation]:
    return read_annotations(dataset_cfg["train_annotation_path"])


def test_rows(annotations: list[Annotation]) -> None:
    assert len(annotations) == 51033


def test_merge_to_coco_annotations(annotations: list[Annotation]) -> None:
    coco = merge_to_coco_annotations(annotations[:1])
    assert len(coco["annotations"]) == 1
    assert len(coco["categories"]) == 1
    assert len(coco["images"]) == 1


def test_create_croped_dataset() -> None:
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "dummy.png",
            },
            {
                "id": 2,
                "file_name": "dummy-1.png",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 100, 200],
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 100, 200],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 100, 200],
            },
        ],
    }
    annotations: list[Any] = [
        {
            "image_file": "dummy.png",
            "species": "species-0",
            "individual_id": "indiviual-0",
        },
        {
            "image_file": "dummy-1.png",
            "species": "species-1",
            "individual_id": "indiviual-1",
        },
    ]
    res = create_croped_dataset(
        coco, annotations, "/app/test_data", "/app/test_outputs"
    )
    assert len(res) == 1
    assert res[0]["image_file"].startswith("dummy")
    assert res[0]["individual_id"] == "indiviual-0"
    assert res[0]["species"] == "species-0"


def test_dataset() -> None:
    dataset = HwadCropedDataset(
        rows=[
            {
                "image_file": "dummy.png",
                "species": "species-0",
                "individual_id": "indiviual-0",
            }
        ],
        image_dir="/app/test_data",
        transform=TrainTransform(dataset_cfg),
    )
    sample = dataset[0]


def test_filter_annotations_by_fold() -> None:

    annotations: list[Any] = [
        {
            "image_file": "img0-box0.png",
            "species": "species-0",
            "individual_id": "indiviual-0",
        },
        {
            "image_file": "img1--b0.png",
            "species": "species-0",
            "individual_id": "indiviual-1",
        },
    ]
    fold = [
        {
            "image": "img0.jpg",
            "individual_id": "indiviual-0",
            "individual_samples": 10,
        },
        {
            "image": "img1.png",
            "individual_id": "indiviual-0",
            "individual_samples": 3,
        },
    ]
    filtered = filter_annotations_by_fold(annotations, fold, min_samples=5)
    assert len(filtered) == 1
    assert filtered[0]["image_file"] == "img0-box0.png"
