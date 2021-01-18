import torch
from object_detection.entities.box import (
    yolo_to_coco,
    YoloBoxes,
    yolo_to_pascal,
    yolo_hflip,
    yolo_vflip,
    box_in_area,
    PascalBoxes,
)
import torch.nn.functional as F
from object_detection.utils import DetectionPlot


def test_yolo_to_coco() -> None:
    w = 100
    h = 200
    yolo = YoloBoxes(
        torch.tensor(
            [
                [0.4, 0.5, 0.2, 0.6],
            ]
        )
    )
    coco = yolo_to_coco(yolo, (w, h))


def test_yolo_to_pascal() -> None:
    w = 100
    h = 200
    yolo = YoloBoxes(
        torch.tensor(
            [
                [0.4, 0.5, 0.2, 0.6],
            ]
        )
    )
    pascal = yolo_to_pascal(yolo, (w, h))
    assert (pascal - torch.tensor([[30, 40, 50, 160]])).abs().sum() < 1e-5


def test_box_in_area() -> None:
    boxes = PascalBoxes(
        torch.tensor(
            [
                [10, 10, 20, 30],
                [20, 20, 30, 30],
                [10, 40, 20, 50],
            ]
        )
    )
    area = torch.tensor([11, 0, 40, 40])
    res = box_in_area(boxes, area)
    assert res.tolist() == [True, True, False]
