import torch
from vnet.box import (
    yolo_to_coco,
    YoloBoxes,
    yolo_to_pascal,
    yolo_hflip,
    yolo_vflip,
    box_in_area,
    box_hflip,
    box_vflip,
    box_padding,
    Boxes,
)
import torch.nn.functional as F
from vnet.utils import DetectionPlot


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
    boxes = Boxes(
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


def test_box_hflip() -> None:
    boxes = Boxes(
        torch.tensor(
            [
                [10, 10, 20, 30],
                [80, 20, 90, 30],
            ]
        )
    )
    img_size = (100, 100)
    res = box_hflip(boxes, img_size)
    assert res.tolist() == [[80, 10, 90, 30], [10, 20, 20, 30]]
    res = box_hflip(res, img_size)
    assert boxes.tolist() == res.tolist()


def test_box_vflip() -> None:
    boxes = Boxes(
        torch.tensor(
            [
                [10, 10, 20, 30],
            ]
        )
    )
    img_size = (100, 100)
    res = box_vflip(boxes, img_size)
    assert res.tolist() == [[10, 70, 20, 90]]
    res = box_vflip(res, img_size)
    assert res.tolist() == boxes.tolist()


def test_box_padding() -> None:
    boxes = Boxes(
        torch.tensor(
            [
                [10, 10, 20, 30],
            ]
        )
    )
    res = box_padding(boxes, 2)
    assert res.tolist() == [[8, 8, 22, 32]]
