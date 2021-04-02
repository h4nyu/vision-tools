import dataclasses, os
from typing import *


@dataclasses.dataclass
class Config:
    root_dir: str = "/store/bms"
    img_size: int = 512

    def __post_init__(self) -> None:
        self.train_csv_path = os.path.join(self.root_dir, "train_labels.csv")

    @property
    def default_transforms(self) -> Any:
        ...
