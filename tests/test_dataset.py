from app.dataset import WheatDataset, CocoDetection, plot_row
from app.preprocess import load_lables
from app import config
from pathlib import Path


def test_plotrow() -> None:
    images = load_lables(limit=10)
    dataset = WheatDataset(images)
    img, annots = dataset[0]
    plot_row(img, annots["boxes"], Path(config.plot_dir).joinpath("test.png"))


def test_coco_detection() -> None:
    dataset = CocoDetection(
        img_folder="/store/coco/val2017",
        ann_file="/store/coco/annotations/instances_val2017.json",
    )
    sample = next(iter(dataset))
    print(sample)
