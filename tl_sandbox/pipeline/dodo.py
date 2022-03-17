from __future__ import annotations

from pprint import pprint

from tl_sandbox import train
from vision_tools.pipeline import CacheAction
from vision_tools.utils import load_config

action = CacheAction()


def task_train() -> dict:
    key = "fold_0_train_model"
    cfg = load_config("../config/baseline.yaml")
    return {
        "targets": [key],
        "file_dep": [],
        "actions": [
            action(
                fn=train,
                kwargs={
                    "cfg": cfg,
                },
                output_kwargs={"rows": "data.pkl"},
            )
        ],
        "verbosity": 2,
    }


def task_preview() -> dict:
    dep = "data.pkl"
    return {
        "file_dep": [dep],
        "actions": [action(pprint, output_args=[dep])],
        "uptodate": [False],
        "verbosity": 2,
    }
