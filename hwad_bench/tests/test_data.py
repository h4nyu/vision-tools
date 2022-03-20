from __future__ import annotations as _

from typing import Any

import pytest

from hwad_bench.data import (
    Annotation,
    HwadCropedDataset,
    TrainTransform,
    add_new_individual,
    create_croped_dataset,
    filter_annotations_by_fold,
    merge_to_coco_annotations,
    read_annotations,
    read_csv,
    save_submission,
    search_threshold,
)
from hwad_bench.models import get_writer
from vision_tools.utils import load_config

dataset_cfg = load_config("config/dataset.yaml")
model_cfg = load_config("config/convnext-base.yaml")

train_cfg = load_config("config/train.yaml")


writer = get_writer(model_cfg)


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
    box_annotations = [
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
    annotations: list[Any] = [
        {
            "image_file": "dummy.png",
            "species": "species-0",
            "individual_id": "indiviual-0",
            "label": 0,
        },
    ]
    res = create_croped_dataset(
        box_annotations=box_annotations,
        annotations=annotations,
        source_dir="/app/test_data",
        dist_dir="/app/test_outputs",
        suffix=".png",
    )
    assert len(res) == 1
    assert res[0]["image_file"].startswith("dummy")
    assert res[0]["individual_id"] == "indiviual-0"
    assert res[0]["species"] == "species-0"


def test_create_test_croped_dataset() -> None:
    box_annotations = [
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
        box_annotations=box_annotations,
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
                "image_file": "/app/datasets/hwad-train-croped-body/0a0cedc8ac6499-89Fx.jpg",
                "species": "species-0",
                "individual_id": "indiviual-0",
                "label": 0,
                "individual_samples": 10,
            }
        ],
        image_dir="/app/test_data",
        transform=TrainTransform(model_cfg),
    )
    for i in range(100):
        sample, _ = dataset[0]
        writer.add_image(f"aug", sample["image"], i)
    writer.flush()


def test_filter_annotations_by_fold() -> None:

    annotations: list[Any] = [
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
    filtered = filter_annotations_by_fold(annotations, fold, min_samples=5)
    assert len(filtered) == 1
    assert filtered[0]["image_file"] == "img0-box0.png"


def test_search_thresold() -> None:
    val_annotations: list[Any] = [
        {
            "image_file": "img0-box0.png",
            "species": "species-0",
            "individual_id": "val-0",
            "label": 1,
        },
        {
            "image_file": "img1--b0.png",
            "species": "species-0",
            "individual_id": "tr-0",
            "label": 2,
        },
    ]
    train_annotations: list[Any] = [
        {
            "image_file": "img0-box0.png",
            "species": "species-0",
            "individual_id": "tr-0",
            "label": 1,
        },
        {
            "image_file": "img1--b0.png",
            "species": "species-0",
            "individual_id": "tr-1",
            "label": 2,
        },
    ]
    submissions: list[Any] = [
        dict(
            image_file="img0-box0.png",
            distances=[0.9, 0.8, 0.7, 0.6, 0.5],
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
        ),
    ]

    res = search_threshold(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        submissions=submissions,
        thresholds=[0.5, 0.95],
    )
    assert res == [{"threshold": 0.5, "score": 0.0}, {"threshold": 0.95, "score": 1.0}]


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


def test_save_submission() -> None:
    submissions: list[Any] = [
        dict(
            image_file="img0-xxxxx.jpg",
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
        )
    ]
    rows = save_submission(submissions, "/app/test_outputs/submissions.csv")
    assert len(rows) == 1
    assert rows[0]["image"] == "img0.jpg"
