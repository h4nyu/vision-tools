import pickle

from torch.utils.data import Dataset

from tl_sandbox import DrawingDataset, Transform, Writer
from vision_tools.utils import load_config

cfg = load_config("/app/tl_sandbox/config/baseline.yaml")

with open("/app/tl_sandbox/pipeline/data.pkl", "rb") as fp:
    annotations = pickle.load(fp)

writer = Writer(cfg)


def test_aug() -> None:
    dataset = DrawingDataset(
        annotations=annotations,
        transform=Transform(cfg),
        image_dir="/app/tl_sandbox/pipeline/images",
    )
    for i in range(1):
        sample, _ = dataset[0]
        writer.add_image(f"aug", sample["image"], i)
