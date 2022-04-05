from __future__ import annotations

from typing import Any

import pytest

from hwad_bench.data import (
    Annotation,
    HwadCropedDataset,
    TrainTransform,
    add_new_individual,
    create_croped_dataset,
    filter_rows_by_fold,
    merge_to_coco_rows,
    read_csv,
    read_rows,
)
from hwad_bench.models import get_writer
from vision_tools.utils import load_config

cfg = load_config("config/experiment.0.yaml")


writer = get_writer(cfg)


@pytest.fixture
def rows() -> list[Annotation]:
    return read_rows(cfg["train_annotation_path"])


def test_rows(rows: list[Annotation]) -> None:
    assert len(rows) == 51033


def test_merge_to_coco_rows(rows: list[Annotation]) -> None:
    coco = merge_to_coco_rows(rows[:1])
    assert len(coco["rows"]) == 1
    assert len(coco["categories"]) == 1
    assert len(coco["images"]) == 1


def test_create_croped_dataset() -> None:
    box_rows = [
        {
            "image_id": "dummy",
            "x1": 20,
            "y1": 30,
            "x2": 40,
            "y2": 200,
            "score": 0.9,
        },
        {
            "image_id": "dummy",
            "x1": 20,
            "y1": 30,
            "x2": 40,
            "y2": 50,
            "score": 0.5,
        },
    ]
    rows: list[Any] = [
        {
            "image_file": "dummy.png",
            "species": "species-0",
            "individual_id": "indiviual-0",
            "label": 0,
        },
    ]
    res = create_croped_dataset(
        box_rows=box_rows,
        rows=rows,
        source_dir="/app/test_data",
        dist_dir="/app/test_outputs",
        suffix=".png",
    )
    assert len(res) == 1
    assert res[0]["image_file"].startswith("dummy")
    assert res[0]["individual_id"] == "indiviual-0"
    assert res[0]["species"] == "species-0"


def test_create_test_croped_dataset() -> None:
    box_rows = [
        {
            "image_id": "dummy",
            "x1": 20,
            "y1": 30,
            "x2": 40,
            "y2": 200,
            "score": 0.9,
        },
        {
            "image_id": "dummy",
            "x1": 20,
            "y1": 30,
            "x2": 40,
            "y2": 50,
            "score": 0.5,
        },
    ]
    res = create_croped_dataset(
        box_rows=box_rows,
        source_dir="/app/test_data",
        dist_dir="/app/test_outputs",
        suffix=".png",
    )
    assert len(res) == 1
    assert res[0]["image_file"].startswith("dummy")
    assert res[0]["individual_id"] is None
    assert res[0]["species"] is None


def test_aug() -> None:
    dataset = HwadCropedDataset(
        rows=[
            {
                "image_file": "/app/datasets/hwad-train-croped/0a0cedc8ac6499.body.jpg",
                "species": "fin_whale",
                "individual_id": "indiviual-0",
                "label": 0,
                "individual_samples": 10,
            }
        ],
        image_dir="/app/test_data",
        transform=TrainTransform(cfg),
    )
    for i in range(20):
        sample, _ = dataset[0]
        writer.add_image(f"aug", sample["image"], i)
        writer.flush()


def test_filter_rows_by_fold() -> None:

    rows: list[Any] = [
        {
            "image_file": "img0-box0.png",
            "species": "species-0",
            "individual_id": "indiviual-0",
            "label": 1,
        },
        {
            "image_file": "img1--b0.png",
            "species": "species-0",
            "individual_id": "indiviual-1",
            "label": 2,
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
    filtered = filter_rows_by_fold(rows, fold)
    assert len(filtered) == 1
    assert filtered[0]["image_file"] == "img0-box0.png"


def test_add_new_individual() -> None:
    submissions: list[Any] = [
        dict(
            image_file="img0.jpg",
            distances=[0.9, 0.8, 0.7, 0.6, 0.5],
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
        ),
    ]

    res = add_new_individual(
        submissions=submissions,
        threshold=0.7,
    )
    assert len(res) == 1
    assert res[0]["individual_ids"][3] == "new_individual"
