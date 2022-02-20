import torch
import torch.nn.functional as F

from vision_tools.box import box_hflip, box_in_area, box_padding, box_vflip


def test_box_in_area() -> None:
    boxes = torch.tensor(
        [
            [10, 10, 20, 30],
            [20, 20, 30, 30],
            [10, 40, 20, 50],
        ]
    )
    area = torch.tensor([11, 0, 40, 40])
    res = box_in_area(boxes, area)
    assert res.tolist() == [True, True, False]


def test_box_hflip() -> None:
    boxes = torch.tensor(
        [
            [10, 10, 20, 30],
            [80, 20, 90, 30],
        ]
    )
    img_size = (100, 100)
    res = box_hflip(boxes, img_size)
    assert res.tolist() == [[80, 10, 90, 30], [10, 20, 20, 30]]
    res = box_hflip(res, img_size)
    assert boxes.tolist() == res.tolist()


def test_box_vflip() -> None:
    boxes = torch.tensor(
        [
            [10, 10, 20, 30],
        ]
    )
    img_size = (100, 100)
    res = box_vflip(boxes, img_size)
    assert res.tolist() == [[10, 70, 20, 90]]
    res = box_vflip(res, img_size)
    assert res.tolist() == boxes.tolist()


def test_box_padding() -> None:
    boxes = torch.tensor(
        [
            [10, 10, 20, 30],
        ]
    )
    res = box_padding(boxes, 2)
    assert res.tolist() == [[8, 8, 22, 32]]
