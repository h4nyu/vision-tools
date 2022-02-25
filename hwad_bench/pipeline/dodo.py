from __future__ import annotations

import json
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import pandas as pd

from hwad_bench.convnext import train
from hwad_bench.data import (
    cleansing,
    create_croped_dataset,
    filter_annotations_by_fold,
    merge_to_coco_annotations,
    read_annotations,
    read_csv,
    read_json,
    summary,
)
from vision_tools.utils import load_config


def persist(key: str, func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        res = func(*args, **kwargs)
        with open(key, "wb") as fp:
            pickle.dump(res, fp)

    return wrapper


def action(
    func: Any,
    args: list[Any] = [],
    kwargs: dict[str, Any] = {},
    output_args: list[str] = [],
    output_kwargs: dict[str, str] = {},
) -> tuple[Any, list, dict]:
    def wrapper() -> Any:
        _kwargs = {}
        _args = []
        for k, v in output_kwargs.items():
            with open(v, "rb") as fp:
                _kwargs[k] = pickle.load(fp)
        for k in output_args:
            with open(k, "rb") as fp:
                _args.append(pickle.load(fp))
        res = func(*[*args, *_args], **{**kwargs, **_kwargs})

    return (wrapper, [], {})


def pkl2json(a: str, b: str) -> None:
    with open(a, "rb") as afp:
        obj = pickle.load(afp)
    with open(b, "wt") as bfp:
        json.dump(obj, bfp)


def task_read_annotations() -> dict:
    key = "train_annotations"
    dataset_cfg = load_config("../config/dataset.yaml")
    file = dataset_cfg["train_annotation_path"]
    return {
        "targets": [key],
        "file_dep": [file],
        "actions": [action(persist(key, read_annotations), [file])],
    }


def task_cleansing() -> dict:
    key = "cleaned_annotations"
    return {
        "targets": [key],
        "file_dep": ["train_annotations"],
        "actions": [
            action(
                persist(key, cleansing),
                output_kwargs={
                    "annotations": "train_annotations",
                },
            )
        ],
    }


def task_summary() -> dict:
    key = "summary"
    return {
        "targets": [key],
        "file_dep": ["cleaned_annotations"],
        "actions": [
            action(
                persist(key, summary),
                output_kwargs={"annotations": "cleaned_annotations"},
            )
        ],
        "verbosity": 2,
    }


def task_merge_to_coco_annotations() -> dict:
    key = "coco_annotations"
    file_deps = ["cleaned_annotations"]
    return {
        "targets": [key],
        "file_dep": file_deps,
        "actions": [
            action(
                persist(key, merge_to_coco_annotations),
                output_args=file_deps,
            )
        ],
        "verbosity": 2,
    }


def task_save_coco_anntation() -> dict:
    key = "coco_annotations.json"
    return {
        "file_dep": ["coco_annotations"],
        "targets": [key],
        "actions": [action(pkl2json, args=["coco_annotations", key])],
    }


def task_read_body_annotation() -> dict:
    key = "coco_body_annotations"
    return {
        "targets": [key],
        "actions": [
            action(
                persist(key, read_json),
                args=["/app/hwad_bench/store/pred-boxes.json"],
            )
        ],
    }


def task_create_croped_dataset() -> dict:
    key = "croped_annotations"
    return {
        "targets": [key],
        "file_dep": ["coco_body_annotations", "cleaned_annotations"],
        "actions": [
            action(
                persist(key, create_croped_dataset),
                output_kwargs={
                    "coco": "coco_body_annotations",
                    "annotations": "cleaned_annotations",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-train",
                    "dist_dir": "/app/datasets/hwad-train-croped-body",
                },
            )
        ],
        "verbosity": 2,
    }


def task_save_croped_annotation() -> dict:
    key = "croped.json"
    dep = "croped_annotations"
    return {
        "targets": [key],
        "file_dep": [dep],
        "actions": [action(pkl2json, args=[dep, key])],
        "verbosity": 2,
    }


def task_read_fold_0_train() -> dict:
    key = "fold_0_train"
    dep = "/app/hwad_bench/store/cv-split/train-fold0.csv"
    return {
        "targets": [key],
        "file_dep": [dep],
        "actions": [
            action(
                persist(key, read_csv),
                args=[dep],
            )
        ],
        "verbosity": 2,
    }


def task_read_fold_0_val() -> dict:
    key = "fold_0_val"
    dep = "/app/hwad_bench/store/cv-split/val-fold0.csv"
    return {
        "targets": [key],
        "file_dep": [dep],
        "actions": [
            action(
                persist(key, read_csv),
                args=[dep],
            )
        ],
        "verbosity": 2,
    }


def task_train_convnext_fold_0() -> dict:
    key = "train_convnext_fold_0"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    train_cfg = load_config("../config/train.yaml")
    return {
        "targets": [key],
        "file_dep": ["croped_annotations", "fold_0_train"],
        "actions": [
            action(
                train,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "annotations": "croped_annotations",
                    "label_map": "label_map",
                    "fold_train": "fold_0_train",
                    "fold_val": "fold_0_val",
                },
            )
        ],
        "verbosity": 2,
    }


# def task_summary() -> dict:
#     dep = "summary"
#     return {
#         "file_dep": [dep],
#         "actions": [action(pprint, output_args=[dep])],
#         "uptodate": [False],
#         "verbosity": 2,
#     }
