from __future__ import annotations as _

from typing import Any

import pytest
from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter("/app/hwad_bench/pipeline/runs/test")


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


def test_train_dataset() -> None:
    dataset = HwadCropedDataset(
        rows=[
            {
                "image_file": "dummy.png",
                "species": "species-0",
                "individual_id": "indiviual-0",
                "label": 0,
            }
        ],
        image_dir="/app/test_data",
        transform=TrainTransform(dataset_cfg),
    )
    for i in range(10):
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
