import torch
from object_detection.models.losses import BoxIoU
from object_detection.models.anchors import Anchors


def test_iou() -> None:
    x_bboxes = torch.tensor([[0, 0, 1, 1]]).float()
    y_bboxes = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3],]).float()

    fn = BoxIoU()
    res = fn(x_bboxes, y_bboxes)
    assert (res - torch.tensor([0.25, 0.0,])).sum() == 0
