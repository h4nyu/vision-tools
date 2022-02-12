from __future__ import annotations
from hwad_bench.data import cleansing
import pandas as pd
from vision_tools.utils import load_config
from hwad_bench.data import read_annotations
from typing import Any, Callable
import pickle
from pathlib import Path


dataset_cfg = load_config('../config/dataset.yaml')['dataset']


def persist(key:str) -> Callable:
    def decorator(func:Any) -> Any:
        def wrapper(*args:Any, **kwargs:Any) -> Any:
            res = func(*args, **kwargs)
            with open(key, 'wb') as fp:
                pickle.dump(res, fp)
        return wrapper
    return decorator

def action(func:Any, args:list[Any]=[], kargs:dict[str,Any]={}, output_args: list[str]=[], output_kargs:dict[str,str] = {}) -> tuple[Any, list, dict]:
    def wrapper(*args:Any, **kwargs:Any) -> Any:
        _kargs = {}
        _args = []
        for k, v in output_kargs.items():
            with open(v, 'rb') as fp:
                _kargs[k] = pickle.load(fp)
        for k in output_args:
            with open(k, 'rb') as fp:
                _args.append(pickle.load(fp))
        res = func(*[*args, *_args], **{**kwargs, **_kargs})
    return (wrapper, args, kargs)



def task_read_annotations() -> dict:
    key = "train_annotations"
    return {
        'targets': [key],
        "actions": [action(persist(key)(read_annotations), [dataset_cfg['train_annotation_path']])],
        'verbosity': 2,
    }

def task_cleansing() -> dict:
    key = "cleaned_annotations"
    return {
        'targets': [key],
        'file_dep': ["train_annotations"],
        "actions": [action(persist(key)(cleansing), output_kargs={
            "annotations": "train_annotations",
        })],
        'verbosity': 2,
    }


def task_preview() -> dict:
    return {
        'file_dep': ["cleaned_annotations"],
        "actions": [action(print, output_args=["cleaned_annotations"])],
        'verbosity': 2,
    }

