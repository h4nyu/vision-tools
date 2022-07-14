import json

import pytorch_lightning as pl
from doit import get_var
from predictor import (
    Config,
    ScoringService,
    Search,
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
cfg = Config.load(get_var("cfg", "./config/v1-0.yaml"))

pl.seed_everything(cfg.seed)


def task_preprocess() -> dict:
    key = "preprocess.cache"
    return {
        "targets": [key],
        "file_dep": [],
        "actions": [
            action(
                key=key,
                fn=preprocess,
                kwargs=dict(
                    image_dir="/app/datasets/train",
                    meta_path="/app/datasets/train_meta.json",
                ),
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


def task_search() -> dict:
    search = Search(n_trials=15, cfg=cfg, fold=action.load("kfold.cache")[cfg.fold])
    return {
        "actions": [
            action(
                fn=search,
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


def task_check_predict() -> dict:
    service = ScoringService
    return {
        "file_dep": ["preprocess.cache", "kfold.cache"],
        "actions": [
            action(
                fn=lambda: ScoringService.get_model,
                kwargs_fn=lambda: dict(
                    model_path="models",
                    reference_path="../datasets/train",
                    reference_meta_path="../datasets/train_meta.json",
                ),
            )
        ],
        "uptodate": [False],
    }
