import dataclasses, os
import dataclasses, os, PIL, albumentations as A
from typing import *


@dataclasses.dataclass
class Config:
    root_dir: str = "/store/bms"

    def __post_init__(self) -> None:
        self.train_csv_path = os.path.join(self.root_dir, "train_labels.csv")
