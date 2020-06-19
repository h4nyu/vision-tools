import typing as t
import pandas as pd
import numpy as np
import torch

from cytoolz.curried import groupby, valmap, pipe, unique, map
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from app import config
from app.entities import Annotations, Boxes
from app.models.utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_xywh_to_xyxy


def parse_boxes(strs: t.List[str], width: float, height: float,) -> Tensor:
    xywh = torch.tensor(np.stack([np.fromstring(s[1:-1], sep=",") for s in strs]))
    xyxy = box_xywh_to_xyxy(xywh)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] / width
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] / height
    return xyxy.float()


def load_lables(limit: t.Optional[int] = None) -> Annotations:
    df = pd.read_csv(config.label_path, nrows=limit)
    rows = df.to_dict("records")
    images = pipe(
        rows,
        groupby(lambda x: x["image_id"]),
        valmap(
            lambda x: Boxes(
                id=x[0]["image_id"],
                w=x[0]["width"],
                h=x[0]["height"],
                fmt="cxcywh",
                boxes=parse_boxes(
                    [b["bbox"] for b in x], x[0]["width"], x[0]["height"]
                ),
            )
        ),
        lambda x: x.values(),
        list,
    )
    return images


class KFold:
    def __init__(self, n_splits: int = config.n_splits):
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.random_state
        )

    def __call__(
        self, annotations: Annotations
    ) -> t.Iterator[t.Tuple[Annotations, Annotations]]:
        rows = annotations
        fold_keys = pipe(rows, map(lambda x: f"{x.boxes.shape[0]}"), list)
        for train, valid in self._skf.split(X=rows, y=fold_keys):
            yield ([rows[i] for i in train], [rows[i] for i in valid])
