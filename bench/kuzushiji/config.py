import os, torch
from vnet.transforms import normalize_mean, normalize_std
import dataclasses


@dataclasses.dataclass
class Config:
    device = torch.device("cuda")
    root_dir = "/store/kuzushiji"
    image_dir = os.path.join(root_dir, "images")
    num_classes = 4787
    image_size = 512 + 256
    n_splits = 5
    log_path = os.path.join(root_dir, "app.log")

    def __post_init__(self) -> None:
        self.code_map_path = os.path.join(self.root_dir, "unicode_translation.csv")
