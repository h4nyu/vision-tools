from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import numpy as np
import pandas as pd
from doit import get_var

from hwad_bench.data import (
    create_croped_dataset,
    filter_rows_by_fold,
    read_csv,
    read_json,
    read_rows,
    save_submission,
    search_threshold,
    summary,
)
from hwad_bench.models import eval_epoch, evaluate, inference, train
from vision_tools.utils import load_config

DOIT_CONFIG = {
    "backend": "json",
    "dep_file": "doit-db.json",
}


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
cfg = load_config(get_var("cfg"))


def pkl2json(a: str, b: str) -> None:
    with open(a, "rb") as afp:
        obj = pickle.load(afp)
    with open(b, "wt") as bfp:
        json.dump(obj, bfp)


def task_read_rows() -> dict:
    key = "train_rows"
    file = cfg["train_annotation_path"]
    return {
        "targets": [key],
        "file_dep": [file],
        "actions": [action(key=key, fn=read_rows, args=[file])],
    }


def task_summary() -> dict:
    key = "summary"
    return {
        "targets": [key],
        "file_dep": ["train_rows"],
        "actions": [
            action(
                key=key,
                fn=summary,
                output_kwargs={"rows": "train_rows"},
            )
        ],
        "verbosity": 2,
    }


def task_read_train_box_rows() -> dict:
    key = "train_box_rows"
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
    key = "train_croped_rows"
    return {
        "targets": [key],
        "file_dep": ["train_box_rows", "train_rows"],
        "actions": [
            action(
                key=key,
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "train_box_rows",
                    "rows": "train_rows",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-train",
                    "dist_dir": "/app/datasets/hwad-train-croped-body",
                },
            )
        ],
        "verbosity": 2,
    }


def task_read_test_box_rows() -> dict:
    key = "test_box_rows"
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
    key = "test_croped_rows"
    return {
        "targets": [key],
        "file_dep": ["test_box_rows"],
        "actions": [
            action(
                key=key,
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "test_box_rows",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-test",
                    "dist_dir": "/app/datasets/hwad-test-croped-body",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_val_rows() -> dict:
    key = "fold_0_val_rows"
    return {
        "targets": [key],
        "file_dep": ["train_croped_rows", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=filter_rows_by_fold,
                output_kwargs={
                    "rows": "train_croped_rows",
                    "fold": "fold_0_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_train_rows() -> dict:
    key = "fold_0_train_rows"
    return {
        "targets": [key],
        "file_dep": [
            "train_croped_rows",
            "fold_0_train",
        ],
        "actions": [
            action(
                key=key,
                fn=filter_rows_by_fold,
                output_kwargs={
                    "rows": "train_croped_rows",
                    "fold": "fold_0_train",
                },
            )
        ],
        "verbosity": 2,
    }


def task_save_croped_annotation() -> dict:
    key = "croped.json"
    dep = "train_croped_rows"
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
    cfg = load_config(get_var("cfg"))
    return {
        "targets": [key],
        "file_dep": [
            "train_croped_rows",
            "fold_0_val_rows",
            "fold_0_train_rows",
        ],
        "actions": [
            action(
                fn=train,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "train_rows": "fold_0_train_rows",
                    "val_rows": "fold_0_val_rows",
                },
            )
        ],
        "verbosity": 2,
    }


def task_eval_epoch() -> dict:
    cfg = load_config(get_var("cfg"))
    return {
        "file_dep": [
            "train_croped_rows",
            "fold_0_val_rows",
            "fold_0_train_rows",
        ],
        "actions": [
            action(
                fn=eval_epoch,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "train_rows": "fold_0_train_rows",
                    "val_rows": "fold_0_val_rows",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_val_submissions() -> dict:
    key = "fold_0_val_submissions"
    return {
        "targets": [key],
        "file_dep": ["train_croped_rows", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=evaluate,
                kwargs={
                    "cfg": cfg,
                    "fold": 0,
                    "image_dir": "/app/datasets/hwad-train-croped-body",
                },
                output_kwargs={
                    "train_rows": "fold_0_train_rows",
                    "val_rows": "fold_0_val_rows",
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
            "fold_0_train_rows",
        ],
        "actions": [
            action(
                key=key,
                fn=search_threshold,
                kwargs={
                    "thresholds": np.linspace(0.0, 1.0, 100).tolist(),
                },
                output_kwargs={
                    "train_rows": "fold_0_train_rows",
                    "val_rows": "fold_0_val_rows",
                    "submissions": "fold_0_val_submissions",
                },
            )
        ],
        "verbosity": 2,
    }


def task_fold_0_submissions() -> dict:
    key = "fold_0_submissions"
    return {
        "targets": [key],
        "file_dep": ["train_croped_rows", "fold_0_train", "fold_0_val"],
        "actions": [
            action(
                key=key,
                fn=inference,
                kwargs={
                    "cfg": cfg,
                    "train_image_dir": "/app/datasets/hwad-train-croped-body",
                    "test_image_dir": "/app/datasets/hwad-test-croped-body",
                },
                output_kwargs={
                    "train_rows": "train_croped_rows",
                    "test_rows": "test_croped_rows",
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
