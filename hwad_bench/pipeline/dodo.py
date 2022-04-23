from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Optional

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
    cv_evaluate,
    cv_registry,
    ensemble_files,
    ensemble_submissions,
    inference,
    post_process,
    preview,
    registry,
    save_submission,
    search_threshold,
    train,
)
from vision_tools.pipeline import CacheAction
from vision_tools.utils import load_config

DOIT_CONFIG = {
    "backend": "json",
    "dep_file": "doit-db.json",
}


action = CacheAction()
cfg = load_config(get_var("cfg", "../config/current.yaml"))
fold = cfg["fold"]
version = cfg["version"]
use_fold = cfg["use_fold"]


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
                    "train_rows": f"train_rows",
                    "body_rows": f"train_body_rows",
                    "fin_rows": f"train_fin_rows",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_cv_registry() -> dict:
    fold = cfg["fold"]
    key = f"cv_registry_{fold}"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=cv_registry,
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


def task_registry() -> dict:
    key = f"registry_{version}"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=registry,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped",
                },
                output_kwargs={
                    "body_rows": f"train_body_rows",
                    "fin_rows": f"train_fin_rows",
                },
            )
        ],
        "verbosity": 2,
    }


def task_cv_evaluate() -> dict:
    fold = cfg["fold"]
    key = f"cv_evaluate_{fold}"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=cv_evaluate,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-train-croped",
                },
                output_kwargs={
                    "body_rows": f"train_body_rows",
                    "fin_rows": f"train_fin_rows",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                    "matcher": f"cv_registry_{fold}",
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
                    "submissions": f"cv_evaluate_{fold}",
                    "fold_train": f"fold_{fold}_train",
                    "fold_val": f"fold_{fold}_val",
                },
            )
        ],
        "verbosity": 2,
    }


def task_inference() -> dict:
    key = f"inference_{version}"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=inference,
                kwargs={
                    "cfg": cfg,
                    "image_dir": "/app/datasets/hwad-test-croped",
                },
                output_kwargs={
                    "train_body_rows": f"train_body_rows",
                    "train_fin_rows": f"train_fin_rows",
                    "test_body_rows": f"test_body_rows",
                    "test_fin_rows": f"test_fin_rows",
                    "matcher": f"registry_{version}",
                },
            )
        ],
        "verbosity": 2,
    }


def task_post_process() -> dict:
    key = f"post_process_{version}"
    return {
        "targets": [key],
        "actions": [
            action(
                key=key,
                fn=post_process,
                kwargs={
                    "cfg": cfg,
                },
                output_kwargs={
                    "train_rows": f"train_rows",
                    "rows": f"inference_{version}",
                },
            )
        ],
        "verbosity": 2,
    }


def task_save_submission() -> dict:
    return {
        "actions": [
            action(
                fn=save_submission,
                kwargs={
                    "output_path": f"{version}.csv",
                },
                output_kwargs={
                    "submissions": f"post_process_{version}",
                },
            )
        ],
        "verbosity": 2,
    }


def task_ensemble_submissions() -> dict:
    return {
        "actions": [
            action(
                key="ensemble",
                fn=ensemble_submissions,
                kwargs={
                    "threshold": 4.0,
                    "output_path": f"ensemble.csv",
                },
                output_kwargs={
                    "submission0": f"inference_v23",
                    "submission1": f"inference_v24",
                    "submission2": f"inference_v25",
                    "train_rows": f"train_rows",
                },
            )
        ],
        "verbosity": 2,
    }


def task_ensemble_files() -> dict:
    key = f"ensemble_files"
    return {
        "actions": [
            action(
                key=key,
                fn=ensemble_files,
                kwargs={
                    "paths": [
                        "/app/hwad_bench/pipeline/v23.csv",
                        "/app/hwad_bench/pipeline/v25.csv",
                        "/app/hwad_bench/pipeline/v24.csv",
                        "/app/hwad_bench/store/submission-coz-fliptta-thr35.csv",
                    ],
                    "output_path": f"ensemble_files.csv",
                },
            )
        ],
        "uptodate": [False],
        "verbosity": 2,
    }


def task_preview() -> dict:
    return {
        "actions": [
            action(
                fn=preview,
                kwargs={"output_path": "/app/test_outputs"},
                output_kwargs={
                    # "submissions": f"post_process_{version}",
                    "submissions": f"ensemble",
                    "train_rows": f"train_rows",
                },
            )
        ],
        "uptodate": [False],
        "verbosity": 2,
    }
