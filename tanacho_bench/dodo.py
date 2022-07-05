import json

from doit import get_var

from tanacho_bench import (
    Config,
    check_folds,
    eda,
    evaluate,
    kfold,
    preprocess,
    preview_dataset,
    train,
)
from vision_tools.pipeline import CacheAction
from vision_tools.utils import load_config

DOIT_CONFIG = {
    "backend": "json",
    "dep_file": "doit-db.json",
    "verbosity": 2,
}


action = CacheAction()
cfg = Config.load(get_var("cfg", "./config/v1.yaml"))
print(cfg.checkpoint_path)


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
        "file_dep": ["preprocess.cache"],
        "actions": [
            action(
                fn=preview_dataset,
                kwargs={"cfg": cfg, "path": "outputs/preview_dataset.png"},
                kwargs_fn=lambda: dict(rows=action.load("preprocess.cache")["rows"]),
            )
        ],
        "uptodate": [False],
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


def task_train() -> dict:
    return {
        "targets": ["train_part.cache"],
        "file_dep": ["kfold.cache"],
        "actions": [
            action(
                fn=train,
                kwargs={"cfg": cfg},
                kwargs_fn=lambda: dict(fold=action.load("kfold.cache")[cfg.fold]),
            )
        ],
    }


def task_evaluate() -> dict:
    return {
        "targets": ["evalulate.cache"],
        "actions": [
            action(
                fn=evaluate,
                kwargs={"cfg": cfg},
                kwargs_fn=lambda: dict(
                    encoders=action.load("preprocess.cache")["encoders"],
                    fold=action.load("kfold.cache")[cfg.fold],
                ),
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
                kwargs_fn=lambda: dict(
                    rows=action.load("preprocess.cache")["rows"],
                ),
            )
        ],
    }


def task_check_folds() -> dict:
    return {
        "file_dep": ["preprocess.cache", "kfold.cache"],
        "actions": [
            action(
                fn=check_folds,
                kwargs_fn=lambda: dict(
                    rows=action.load("preprocess.cache")["rows"],
                    folds=action.load("kfold.cache"),
                ),
            )
        ],
        "uptodate": [False],
    }
