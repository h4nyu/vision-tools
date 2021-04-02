import dataclasses, os, PIL, albumentations as A, numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from bench.bms.config import Config
from torch.utils.data import Dataset
from typing import *
from vnet import Image
from bench.bms.config import Config
from joblib import Memory

memory = Memory("/tmp", verbose=0)

INCHI_PREFIX = "InChI="


@dataclasses.dataclass
class InChI:
    value: str

    def __post_init__(self) -> None:
        self.value = self.value.lstrip(INCHI_PREFIX)
        layers = self.value.split("/")
        self.name = layers[1]
        self.connections = layers[2].split(',')
        self.num_layers = len(layers)



@dataclasses.dataclass
class Row:
    id: str
    root_dir: str
    inchi_value: str

    def __post_init__(self) -> None:
        self.image_path = os.path.join(
            self.root_dir,
            self.id[0],
            self.id[1],
            self.id[2],
            f"{self.id}.png",
        )
        self.inchi = InChI(self.inchi_value)


@memory.cache
def read_train_rows(config: Config, nrows: Optional[int] = None) -> List[Row]:
    df = pd.read_csv(config.train_csv_path, nrows=nrows)
    rows: List[Row] = []
    for (_, csv_row) in df.iterrows():
        row = Row(
            id=csv_row["image_id"],
            root_dir=config.root_dir,
            inchi_value=csv_row["InChI"],
        )
        rows.append(row)
    return rows


class BMSDataset(Dataset):
    def __init__(
        self,
        rows: List[Row],
        transforms: Any,
    ) -> None:
        self.rows = rows
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Image,]:
        ...
        # row = self.rows[idx]
        # pil_img = PIL.Image.open(row.image_path)
        # img_arr = np.array(pil_img)
        # # return (img_arr,)


@dataclasses.dataclass
class DataConfig(Config):
    image_size: int = 512

    def __post_init__(self) -> None:
        self.train_csv_path = os.path.join(self.root_dir, "train_labels.csv")

    @property
    def default_transforms(self) -> Any:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=0,
                ),
                ToTensorV2(),
            ],
        )

    @property
    def train_dataset(self) -> BMSDataset:
        rows = read_train_rows(self)
        return BMSDataset(rows=rows, transforms=self.default_transforms)
