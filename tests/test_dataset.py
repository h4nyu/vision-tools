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
    dataset = WheatDataset(images)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
    visualize = VisualizeHeatmap(Path("/store/plot"))
    preprocess = PreProcess()

    samples, targets = next(iter(dataloader))
    samples, targets = preprocess((samples, targets))
    visualize(samples, targets, targets)
    #  img, annots = dataset[0]
    #  plot = DetectionPlot()
    #  plot.with_image(img)
    #  plot.with_boxes(annots["boxes"], color="red")
    #  plot.save(str(Path(config.plot_dir).joinpath(f"test-{i}.png")))
