from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import pandas as pd

from hwad_bench.convnext import create_train_matcher, evaluate, train
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


class CacheAction:
    def _persist(self, key: str, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            res = func(*args, **kwargs)
            with open(key, "wb") as fp:
                pickle.dump(res, fp)

        return wrapper

    def __call__(
        self,
        fn: Any,
        args: list[Any] = [],
        kwargs: dict[str, Any] = {},
        output_args: list[str] = [],
        output_kwargs: dict[str, str] = {},
        key: str = None,
    ) -> tuple[Any, list, dict]:
        if key is not None:
            fn = self._persist(key, fn)

        def wrapper() -> Any:
            _kwargs = {}
            _args = []
            for k, v in output_kwargs.items():
                with open(v, "rb") as fp:
                    _kwargs[k] = pickle.load(fp)
            for v in output_args:
                with open(v, "rb") as fp:
                    _args.append(pickle.load(fp))
            res = fn(*[*args, *_args], **{**kwargs, **_kwargs})

        return (wrapper, [], {})


action = CacheAction()


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
        "actions": [action(key=key, fn=read_annotations, args=[file])],
    }


def task_cleansing() -> dict:
    key = "cleaned_annotations"
    return {
        "targets": [key],
        "file_dep": ["train_annotations"],
        "actions": [
            action(
                cleansing,
                output_kwargs={
                    "annotations": "train_annotations",
                },
                key=key,
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
                key=key,
                fn=summary,
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
                key=key,
                fn=merge_to_coco_annotations,
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


def task_read_train_box_annotations() -> dict:
    key = "train_box_annotations"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=read_csv,
                args=["/app/hwad_bench/store/pred-r7-st.csv"],
            )
        ],
    }


def task_create_train_croped_dataset() -> dict:
    key = "train_croped_annotations"
    return {
        "targets": [key],
        "file_dep": ["train_box_annotations", "cleaned_annotations"],
        "actions": [
            action(
                key=key,
                fn=create_croped_dataset,
                output_kwargs={
                    "box_annotations": "train_box_annotations",
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


def task_read_test_box_annotations() -> dict:
    key = "test_box_annotations"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=read_csv,
                args=["/app/hwad_bench/store/pred-test.csv"],
            )
        ],
    }


def task_create_test_croped_dataset() -> dict:
    key = "test_croped_annotations"
    return {
        "targets": [key],
        "file_dep": ["test_box_annotations"],
        "actions": [
            action(
                key=key,
                fn=create_croped_dataset,
                output_kwargs={
                    "box_annotations": "test_box_annotations",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-test",
                    "dist_dir": "/app/datasets/hwad-test-croped-body",
                },
            )
        ],
        "verbosity": 2,
    }


def task_save_croped_annotation() -> dict:
    key = "croped.json"
    dep = "train_croped_annotations"
    return {
        "targets": [key],
        "file_dep": [dep],
        "actions": [action(fn=pkl2json, args=[dep, key])],
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
                key=key,
                fn=read_csv,
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
                key=key,
                fn=read_csv,
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
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                fn=train,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "annotations": "train_croped_annotations",
                    "fold_train": "fold_0_train",
                    "fold_val": "fold_0_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_train_matcher_fold_0() -> dict:
    key = "train_matcher_fold_0"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    train_cfg = load_config("../config/train.yaml")
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                fn=create_train_matcher,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "annotations": "train_croped_annotations",
                    "fold_train": "fold_0_train",
                    "fold_val": "fold_0_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_evaluate_convnext_fold_0() -> dict:
    key = "evaluate_convnext_fold_0"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    train_cfg = load_config("../config/train.yaml")
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                fn=evaluate,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "annotations": "train_croped_annotations",
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
