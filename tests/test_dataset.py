import torch
from app.dataset import WheatDataset
from app.preprocess import load_lables
from app import config
from pathlib import Path
from app.utils import DetectionPlot


def test_plotrow() -> None:
    images = load_lables()
    dataset = WheatDataset(images)
    for i in range(10):
        img, annots = dataset[0]
        plot = DetectionPlot()
        plot.with_image(img)
        plot.with_boxes(annots["boxes"], color="red")
        plot.save(str(Path(config.plot_dir).joinpath(f"test-{i}.png")))
