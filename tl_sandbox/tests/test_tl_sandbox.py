import pickle

from torch.utils.data import Dataset

from tl_sandbox import DrawingDataset, Transform, Writer, kfold
from vision_tools.utils import load_config

cfg = load_config("/app/tl_sandbox/config/baseline.yaml")

with open("/app/tl_sandbox/pipeline/data.pkl", "rb") as fp:
    rows = pickle.load(fp)

writer = Writer(cfg)


def test_aug() -> None:
    dataset = DrawingDataset(
        rows=rows,
        transform=Transform(cfg),
        image_dir="/app/tl_sandbox/pipeline/images",
    )
    for i in range(1):
        sample, _ = dataset[0]
        writer.add_image(f"aug", sample["image"], i)


def test_kfold() -> None:
    tra, val = kfold(rows, 5, 0)
    assert len(tra) == 80
    assert len(val) == 20
