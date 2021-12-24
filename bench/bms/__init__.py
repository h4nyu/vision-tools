import os, torch
from vision_tools.transforms import normalize_mean, normalize_std
import dataclasses


@dataclasses.dataclass
class Config:
    device = torch.device("cuda")
    root_dir = "/store/bms"
    image_dir = os.path.join(root_dir, "images")
    n_splits = 5
    log_path = os.path.join(root_dir, "app.log")
