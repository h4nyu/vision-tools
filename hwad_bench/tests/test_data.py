from __future__ import annotations as _
import pytest
from typing import Any
from hwad_bench.data import read_annotations, Annotation, merge_to_coco_annotations
from vision_tools.utils import load_config

dataset_cfg = load_config("config/dataset.yaml")["dataset"]


@pytest.fixture
def annotations() -> list[Annotation]:
    return read_annotations(dataset_cfg["train_annotation_path"])


def test_rows(annotations: list[Annotation]) -> None:
    assert len(annotations) == 51033


def test_merge_to_coco_annotations(annotations: list[Annotation]) -> None:
    coco_images: list[Any] = [
        {
            "id": 1,
            "file_name": "00021adfb725ed.jpg",
            "width": 1280,
            "height": 720,
        },
    ]
    coco = merge_to_coco_annotations(annotations[:1], coco_images)
    assert len(coco["annotations"]) == 1
    assert len(coco["categories"]) == 1
    assert len(coco["images"]) == 1
