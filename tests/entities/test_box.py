import torch
from object_detection.entities.box import yolo_to_coco, YoloBoxes, yolo_to_pascal
import torch.nn.functional as F


def test_yolo_to_coco() -> None:
    w = 100
    h = 200
    yolo = YoloBoxes(torch.tensor([[0.4, 0.5, 0.2, 0.6],]))
    coco = yolo_to_coco(yolo, (w, h))
    print(coco)

def test_yolo_to_pascal() -> None:
    w = 100
    h = 200
    yolo = YoloBoxes(torch.tensor([[0.4, 0.5, 0.2, 0.6],]))
    pascal = yolo_to_pascal(yolo, (w, h))
    assert (pascal - torch.tensor([[30, 40, 50, 160]])).sum() == 0
