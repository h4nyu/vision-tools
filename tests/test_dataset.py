from app.dataset import WheatDataset, CocoDetection
from app.preprocess import load_lables


def test_wheat() -> None:
    images = load_lables(limit=10)
    dataset = WheatDataset(images)
    img, target = dataset[0]
    print(img.shape)
    print(target)


def test_coco_detection() -> None:
    dataset = CocoDetection(
        img_folder="/store/coco/val2017",
        ann_file="/store/coco/annotations/instances_val2017.json",
    )
    sample = next(iter(dataset))
    print(sample)
