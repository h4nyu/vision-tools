from __future__ import annotations
from hwad_bench.data import cleansing
import pandas as pd
import json
from vision_tools.utils import load_config
from hwad_bench.data import (
    read_annotations,
    summary,
    merge_to_coco_annotations,
    create_croped_dataset,
    read_json,
)
from typing import Any, Callable
import pickle
from pathlib import Path
from pprint import pprint


dataset_cfg = load_config("../config/dataset.yaml")["dataset"]


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
                args=["/app/hwad_bench/store/hwad-train-labeling-r3-20.json"],
            )
        ],
    }


def task_create_croped_dataset() -> dict:
    key = "croped_dataset"
    return {
        "targets": [key],
        "actions": [
            action(
                persist(key, create_croped_dataset),
                output_kwargs={
                    "coco": "coco_body_annotations",
                },
                kwargs={
                    "source_dir": "/app/datatests/hwad-train-labeling-r3",
                    "dist_dir": "/app/hwad_bench/store/hwad-train-croped-body",
                },
            )
        ],
        "verbosity": 2,
    }


# def task_preview() -> dict:
#     output_key = "summary"
#     return {
#         "file_dep": [output_key],
#         "actions": [action(pprint, output_args=[output_key])],
#         "uptodate": [False],
#         "verbosity": 2,
#     }
