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
    summary,
)
from hwad_bench.models import (
    evaluate,
    inference,
    save_submission,
    search_threshold,
    train,
)
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
cfg = load_config(get_var("cfg", "../config/experiment.0.yaml"))
fold = cfg["fold"]


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


def task_read_boxes() -> dict:
    return {
        "targets": [
            "train_body_boxes",
            "train_fin_boxes",
            "test_body_boxes",
            "test_fin_boxes",
        ],
        "actions": [
            action(
                key="train_body_boxes",
                fn=read_csv,
                args=["/app/hwad_bench/store/pred-r7-st.csv"],
            ),
            action(
                key="train_fin_boxes",
                fn=read_csv,
                args=["/app/hwad_bench/store/train-backfin-boxes.csv"],
            ),
            action(
                key="test_body_boxes",
                fn=read_csv,
                args=["/app/hwad_bench/store/pred-test.csv"],
            ),
            action(
                key="test_fin_boxes",
                fn=read_csv,
                args=["/app/hwad_bench/store/test-backfin-boxes.csv"],
            ),
        ],
    }


def task_croped_dataset() -> dict:
    return {
        "targets": [
            "train_body_rows",
            "train_fin_rows",
            "test_body_rows",
            "test_fin_rows",
        ],
        "file_dep": [
            "train_rows",
            "train_body_boxes",
            "train_fin_boxes",
            "test_body_boxes",
            "test_fin_boxes",
        ],
        "actions": [
            action(
                key="train_body_rows",
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "train_body_boxes",
                    "rows": "train_rows",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-train",
                    "dist_dir": "/app/datasets/hwad-train-croped",
                    "part": "body",
                },
            ),
            action(
                key="train_fin_rows",
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "train_fin_boxes",
                    "rows": "train_rows",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-train",
                    "dist_dir": "/app/datasets/hwad-train-croped",
                    "part": "fin",
                },
            ),
            action(
                key="test_body_rows",
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "test_body_boxes",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-test",
                    "dist_dir": "/app/datasets/hwad-test-croped",
                    "part": "body",
                },
            ),
            action(
                key="test_fin_rows",
                fn=create_croped_dataset,
                output_kwargs={
                    "box_rows": "test_fin_boxes",
                },
                kwargs={
                    "source_dir": "/app/datasets/hwad-test",
                    "dist_dir": "/app/datasets/hwad-test-croped",
                    "part": "fin",
                },
            ),
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


def task_save_croped_annotation() -> dict:
    key = "croped.json"
    dep = "train_croped_rows"
    return {
        "targets": [key],
        "file_dep": [dep],
        "actions": [action(fn=pkl2json, args=[dep, key])],
        "verbosity": 2,
    }


def task_read_fold() -> dict:
    num_split = cfg["num_split"]
    return {
        "targets": [
            *[f"fold_{i}_train" for i in range(num_split)],
            *[f"fold_{i}_val" for i in range(num_split)],
        ],
        "actions": [
            *[
                action(
                    key=f"fold_{i}_train",
                    fn=read_csv,
                    args=[f"/app/hwad_bench/store/cv-split/train-fold{i}.csv"],
                )
                for i in range(num_split)
            ],
            *[
                action(
                    key=f"fold_{i}_val",
                    fn=read_csv,
                    args=[f"/app/hwad_bench/store/cv-split/val-fold{i}.csv"],
                )
                for i in range(num_split)
            ],
        ],
        "verbosity": 2,
    }


def task_train_model() -> dict:
    fold = cfg["fold"]
    return {
        "actions": [
            action(
                fn=train,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped",
                },
                output_kwargs={
                    "body_rows": f"train_body_rows",
                    "fin_rows": f"train_fin_rows",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_val_submissions() -> dict:
    fold = cfg["fold"]
    key = f"fold_{fold}_val_submissions"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=evaluate,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped",
                },
                output_kwargs={
                    "body_rows": f"train_body_rows",
                    "fin_rows": f"train_fin_rows",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_search_threshold() -> dict:
    fold = cfg["fold"]
    key = f"fold_{fold}_search_threshold"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=search_threshold,
                kwargs={
                    "thresholds": np.linspace(0.0, 1.0, 100).tolist(),
                },
                output_kwargs={
                    "submissions": f"fold_{fold}_val_submissions",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_submission() -> dict:
    key = f"fold_{fold}_submission"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=inference,
                kwargs={
                    "cfg": cfg,
                    "train_image_dir": "/app/datasets/hwad-train-croped",
                    "test_image_dir": "/app/datasets/hwad-test-croped",
                    "threshold": cfg.get("threshold", None),
                },
                output_kwargs={
                    "train_body_rows": f"train_body_rows",
                    "train_fin_rows": f"train_fin_rows",
                    "test_body_rows": f"test_body_rows",
                    "test_fin_rows": f"test_fin_rows",
                    "search_thresholds": "fold_0_search_threshold",
                },
            )
        ],
        "verbosity": 2,
    }


def task_save_submission() -> dict:
    key = f"fold_{fold}_submission_csv"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=save_submission,
                kwargs={
                    "output_path": f"fold_{fold}_submission.csv",
                },
                output_kwargs={
                    "submissions": f"fold_{fold}_submission",
                },
            )
        ],
        "verbosity": 2,
    }


def task_preview() -> dict:
    dep = "fold_0_search_threshold"
    return {
        "actions": [action(pprint, output_args=[dep])],
        "uptodate": [False],
        "verbosity": 2,
    }
