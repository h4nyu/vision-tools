import pickle

from torch.utils.data import Dataset

from tl_sandbox import DrawingDataset, Transform
from vision_tools.utils import load_config

cfg = load_config("/app/tl_sandbox/config/baseline.yaml")

with open("/app/tl_sandbox/pipeline/data.pkl", "rb") as fp:
    annotations = pickle.load(fp)


def test_aug() -> None:
    dataset = DrawingDataset(
        annotations=annotations,
        transform=Transform(cfg),
        image_dir="/app/tl_sandbox/pipeline/images",
    )
