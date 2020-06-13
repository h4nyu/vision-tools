from app.dataset import WheatDataset, CocoDetection, plot_row
from app.preprocess import load_lables
from app import config
from pathlib import Path
from app.utils import DetectionPlot


def test_plotrow() -> None:
    images = load_lables(limit=10)
    dataset = WheatDataset(images)
    img, annots = dataset[0]
    plot = DetectionPlot()
    plot.with_image(img)
    plot.with_boxes(annots["boxes"], color="red")
    plot.save(str(Path(config.plot_dir).joinpath("test.png")))
    #  print(img.shape)
    #  plot.with_open
    #  plot_row(img, , )


def test_coco_detection() -> None:
    dataset = CocoDetection(
        img_folder="/store/coco/val2017",
        ann_file="/store/coco/annotations/instances_val2017.json",
    )
    sample = next(iter(dataset))
    print(sample)
