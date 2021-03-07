import torch, os
from torch.utils.data import Dataset
from vnet import Image, Boxes, Labels
from dataclasses import dataclass
from typing import Any, TypedDict
import pandas as pd
import cytoolz as tlz
from joblib import Memory

location = '/tmp'
memory = Memory(location, verbose=0)


Row = TypedDict("Row", {"id": str, "image_id": str, "boxes": Boxes, "labels": Labels})


def read_code_map(fp: str) -> dict[str, int]:
    df = pd.read_csv(fp)
    label_map: dict[str, int] = dict()
    for id, csv_row in df.iterrows():
        code, value = csv_row
        label_map[code] = id

    return label_map

@memory.cache
def read_rows(root_dir: str) -> list[Row]:
    row_path = os.path.join(root_dir, "train.csv")
    code_path = os.path.join(root_dir, "unicode_translation.csv")
    codes = read_code_map(code_path)
    df = pd.read_csv(row_path)
    rows:list[Row] = []
    for id, csv_row in df.iterrows():
        labels = []
        boxes = []
        for code, x0, y0, x1, y1 in tlz.partition(5, csv_row["labels"].split(" ")):
            labels.append(codes[code])
            boxes.append(
                [
                    float(x0),
                    float(y0),
                    float(x1),
                    float(y1),
                ]
            )
        row:Row = dict(
            id=id,
            image_id=csv_row["image_id"],
            boxes=Boxes(torch.tensor(boxes)),
            labels=Labels(torch.tensor(labels)),
        )
        rows.append(row)

    return rows


class KuzushijiDataset(Dataset):
    def __init__(
        self,
        image_dir: str = "/store/images",
        transforms: Any = None,
    ) -> None:
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[str, Image, Boxes, Labels]:
        ...
