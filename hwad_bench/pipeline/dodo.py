from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import numpy as np
import pandas as pd

from hwad_bench.data import (
    create_croped_dataset,
    filter_annotations_by_fold,
    read_annotations,
    read_csv,
    read_json,
    save_submission,
    search_threshold,
    summary,
)
from hwad_bench.models import evaluate, inference, train
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


def task_summary() -> dict:
    key = "summary"
    return {
        "targets": [key],
        "file_dep": ["train_annotations"],
        "actions": [
            action(
                key=key,
                fn=summary,
                output_kwargs={"annotations": "train_annotations"},
            )
        ],
        "verbosity": 2,
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
        "file_dep": ["train_box_annotations", "train_annotations"],
        "actions": [
            action(
                key=key,
                fn=create_croped_dataset,
                output_kwargs={
                    "box_annotations": "train_box_annotations",
                    "annotations": "train_annotations",
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


def task_fold_0_val_annotations() -> dict:
    key = "fold_0_val_annotations"
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=filter_annotations_by_fold,
                output_kwargs={
                    "annotations": "train_croped_annotations",
                    "fold": "fold_0_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_train_annotations() -> dict:
    key = "fold_0_train_annotations"
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train"],
        "actions": [
            action(
                key=key,
                fn=filter_annotations_by_fold,
                output_kwargs={
                    "annotations": "train_croped_annotations",
                    "fold": "fold_0_train",
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


def task_fold_0_train_model() -> dict:
    key = "fold_0_train_model"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                fn=train,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "train_annotations": "fold_0_train_annotations",
                    "val_annotations": "fold_0_val_annotations",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_val_submissions() -> dict:
    key = "fold_0_val_submissions"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=evaluate,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "train_annotations": "fold_0_train_annotations",
                    "val_annotations": "fold_0_val_annotations",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_search_threshold() -> dict:
    key = "fold_0_search_threshold"
    return {
        "targets": [key],
        "file_dep": [
            "fold_0_val_submissions",
            "fold_0_train_annotations",
        ],
        "actions": [
            action(
                key=key,
                fn=search_threshold,
                kwargs={
                    "thresholds": np.linspace(0.0, 1.0, 100).tolist(),
                },
                output_kwargs={
                    "train_annotations": "fold_0_train_annotations",
                    "val_annotations": "fold_0_val_annotations",
                    "submissions": "fold_0_val_submissions",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_submissions() -> dict:
    key = "fold_0_submissions"
    dataset_cfg = load_config("../config/dataset.yaml")
    model_cfg = load_config("../config/convnext-base.yaml")
    return {
        "targets": [key],
        "file_dep": ["train_croped_annotations", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=inference,
                kwargs={
                    "dataset_cfg": dataset_cfg,
                    "model_cfg": model_cfg,
                    "train_image_dir": "/app/datasets/hwad-train-croped-body",
                    "test_image_dir": "/app/datasets/hwad-test-croped-body",
                },
                output_kwargs={
                    "train_annotations": "train_croped_annotations",
                    "test_annotations": "test_croped_annotations",
                    "search_thresholds": "fold_0_search_threshold",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_submission_csv() -> dict:
    key = "fold_0_submission_csv"
    return {
        "targets": [key],
        "file_dep": ["fold_0_submissions"],
        "actions": [
            action(
                key=key,
                fn=save_submission,
                kwargs={
                    "output_path": "fold_0_submission.csv",
                },
                output_kwargs={
                    "submissions": "fold_0_submissions",
                },
            )
        ],
        "verbosity": 2,
    }


def task_preview() -> dict:
    dep = "fold_0_search_threshold"
    return {
        "file_dep": [dep],
        "actions": [action(pprint, output_args=[dep])],
        "uptodate": [False],
        "verbosity": 2,
    }
