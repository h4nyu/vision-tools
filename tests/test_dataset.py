import torch
from app.dataset import WheatDataset
from app.preprocess import load_lables
from app.models.centernet import VisualizeHeatmap, PreProcess
from torch.utils.data import DataLoader
from app import config
from pathlib import Path
from app.utils import DetectionPlot
from app.dataset import collate_fn


def test_plotrow() -> None:
    images = load_lables()
    dataset = WheatDataset(images,)
    for i in range(5):
        img, annots = dataset[1]
        plot = DetectionPlot(figsize=(6, 6))
        plot.with_image(img)
        plot.with_boxes(annots["boxes"], color="red")
        plot.save(str(Path(config.plot_dir).joinpath(f"test-{i}.png")))
