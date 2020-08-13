import torch
import torch.nn.functional as F
from object_detection.models.losses import DIoU, GIoU
from object_detection import PascalBoxes
from torch import nn, Tensor


def test_diou() -> None:
    fn = DIoU()
    pred_boxes = PascalBoxes(
        torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3],])
    )
    tgt_boxes = PascalBoxes(torch.tensor([[0.2, 0.2, 0.3, 0.3]]))
    res = fn(pred_boxes, tgt_boxes,)
    assert res.shape == (2, 1)
    assert F.l1_loss(res, torch.tensor([[1.25], [0.0]])) < 1e-7


def test_giou() -> None:
    fn = GIoU()
    pred_boxes = PascalBoxes(
        torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.3, 0.3],])
    )
    tgt_boxes = PascalBoxes(torch.tensor([[0.2, 0.2, 0.3, 0.3]]))
    res = fn(pred_boxes, tgt_boxes,)
    assert res.shape == (2, 1)
    assert F.l1_loss(res, torch.tensor([[1.5], [1.25]])) < 1e-7
