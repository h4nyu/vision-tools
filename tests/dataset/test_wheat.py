import torch
from app.dataset.wheat import WheatDataset
from app.models.centernet import PreProcess
from torch.utils.data import DataLoader
from app import config
from pathlib import Path
from app.utils import DetectionPlot
from app import config


def test_plotrow() -> None:
    dataset = WheatDataset(config.annot_file,)
    image_id, img, boxes = dataset[0]
    assert img.dtype == torch.float32
    assert boxes.dtype == torch.float32

    for i in range(5):
        _, img, boxes = dataset[i]
        plot = DetectionPlot(figsize=(6, 6))
        plot.with_image(img)
        plot.with_yolo_boxes(boxes, color="red")
        plot.save(f"{config.working_dir}/test-dataset-{i}.png")
