import torch
from object_detection.entities.box import (
    yolo_to_coco,
    YoloBoxes,
    yolo_to_pascal,
    yolo_hflip,
    yolo_vflip,
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
    print(coco)


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


def test_yolo_hflip() -> None:
    in_boxes = YoloBoxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.05],
            ]
        )
    )
    out_boxes = yolo_hflip(in_boxes)
    assert (out_boxes - torch.tensor([[0.9, 0.1, 0.1, 0.05]])).abs().sum() == 0
    plot = DetectionPlot(w=100, h=100)
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-yolo-hflip.png")


def test_yolo_vflip() -> None:
    in_boxes = YoloBoxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.05],
            ]
        )
    )
    out_boxes = yolo_vflip(in_boxes)
    assert (out_boxes - torch.tensor([[0.1, 0.9, 0.1, 0.05]])).abs().sum() == 0
    plot = DetectionPlot(w=100, h=100)
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes, color="red")
    plot.save(f"store/test-yolo-vflip.png")
