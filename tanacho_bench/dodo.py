import json

from doit import get_var

from tanacho_bench import eda, kfold, preprocess, preview_dataset, train_category
from vision_tools.pipeline import CacheAction
from vision_tools.utils import load_config

DOIT_CONFIG = {
    "backend": "json",
    "dep_file": "doit-db.json",
    "verbosity": 2,
}


action = CacheAction()
cfg = load_config(get_var("cfg", "./config/v1.yaml"))


def task_preprocess() -> dict:
    key = "preprocess.cache"
    return {
        "targets": [key],
        "file_dep": [],
        "actions": [
            action(
                key=key,
                fn=preprocess,
                args=["/app/datasets/train_meta.json", "/app/datasets/train"],
            )
        ],
    }


def task_preview_dataset() -> dict:
    return {
        "targets": ["preview_dataset"],
        "file_dep": ["train_rows.cache"],
        "actions": [
            action(
                fn=preview_dataset,
                kwargs={"cfg": cfg, "path": "outputs/preview_dataset.png"},
                kwargs_fn=lambda: dict(rows=action.load("preprocess.cache")["rows"]),
            )
        ],
    }


def task_kfold() -> dict:
    key = "kfold.cache"
    return {
        "targets": [key],
        "file_dep": ["preprocess.cache"],
        "actions": [
            action(
                key=key,
                fn=kfold,
                kwargs={"cfg": cfg},
                kwargs_fn=lambda: dict(rows=action.load("preprocess.cache")["rows"]),
            )
        ],
    }


def task_train_category() -> dict:
    return {
        "targets": ["train.cache"],
        "file_dep": ["kfold.cache"],
        "actions": [
            action(
                fn=train_category,
                kwargs={"cfg": cfg},
                kwargs_fn=lambda: dict(fold=action.load("kfold.cache")[cfg["fold"]]),
            )
        ],
    }


def task_eda() -> dict:
    return {
        "targets": ["eda"],
        "file_dep": ["preprocess.cache"],
        "actions": [
            action(
                fn=eda,
                kwargs_fn=lambda: dict(rows=action.load("preprocess.cache")["rows"]),
            )
        ],
    }
