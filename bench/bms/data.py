import dataclasses, os, PIL, albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from bench.bms.config import Config
from torch.utils.data import Dataset
from typing import *
from vnet import Image


@dataclasses.dataclass
class Row:
    id: str
    root_dir: str

    def __post_init__(self) -> None:
        self.image_path = os.path.join(
            self.root_dir,
            self.id[0],
            self.id[1],
            self.id[2],
            f"{self.id}.png",
        )


def read_train_rows(config: Config, nrows: Optional[int] = None) -> List[Row]:
    df = pd.read_csv(config.train_csv_path, nrows=nrows)
    rows: List[Row] = []
    for (_, csv_row) in df.iterrows():
        row = Row(
            id=csv_row["image_id"],
            root_dir=config.root_dir,
        )
        rows.append(row)
    return rows


default_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size, min_width=config.image_size, border_mode=0
        ),
        normalize,
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class BMSDataset(Dataset):
    def __init__(
        self,
        rows: List[Row],
        transforms: Any = None,
    ) -> None:
        self.rows = rows
        self.transforms = default_transforms if transforms is None else transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Image,]:
        row = self.rows[idx]
        pil_img = PIL.Image.open(row.image_path)
        img_arr = np.array(pil_img)
        return (img,)
