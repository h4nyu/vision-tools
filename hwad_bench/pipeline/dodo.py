from __future__ import annotations
from hwad_bench.data import cleansing
import pandas as pd
from vision_tools.utils import load_config
from hwad_bench.data import read_annotations, summary
from typing import Any, Callable
import pickle
from pathlib import Path
from pprint import pprint


dataset_cfg = load_config('../config/dataset.yaml')['dataset']


def persist(key:str, func:Callable) -> Callable:
    def wrapper(*args:Any, **kwargs:Any) -> Any:
        res = func(*args, **kwargs)
        with open(key, 'wb') as fp:
            pickle.dump(res, fp)
    return wrapper

def action(func:Any, args:list[Any]=[], kwargs:dict[str,Any]={}, output_args: list[str]=[], output_kargs:dict[str,str] = {}) -> tuple[Any, list, dict]:
    def wrapper() -> Any:
        _kargs = {}
        _args = []
        for k, v in output_kargs.items():
            with open(v, 'rb') as fp:
                _kargs[k] = pickle.load(fp)
        for k in output_args:
            with open(k, 'rb') as fp:
                _args.append(pickle.load(fp))
        res = func(*[*args, *_args], **{**kwargs, **_kargs})
    return (wrapper, [], {})



def task_read_annotations() -> dict:
    key = "train_annotations"
    file = dataset_cfg['train_annotation_path']
    return {
        'targets': [key],
        'file_dep': [file],
        "actions": [action(persist(key, read_annotations), [file])],
    }

def task_cleansing() -> dict:
    key = "cleaned_annotations"
    return {
        'targets': [key],
        'file_dep': ["train_annotations"],
        "actions": [action(persist(key, cleansing), output_kargs={
            "annotations": "train_annotations",
        })],
    }


def task_summary() -> dict:
    key = "summary"
    return {
        'targets': [key],
        'file_dep': ["cleaned_annotations"],
        "actions": [action(persist(key, summary), output_kargs={
            "annotations": "cleaned_annotations"
        })],
    }

def task_preview() -> dict:
    return {
        'file_dep': ["summary"],
        "actions": [action(pprint, output_args=["summary"])],
        'uptodate': [False],
        'verbosity': 2,
    }
