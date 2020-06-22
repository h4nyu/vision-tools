import torch
from torch import nn
from app.models.detr import DETR
from app.models.utils import NestedTensor


def test_detr_init_box() -> None:
    ...
    #  imgs = torch.rand(1, 3, 1024, 1024)
    #  b, _, w, h = imgs.shape
    #  msks = torch.zeros((b, w, h), dtype=torch.bool)
    #  inputs = NestedTensor(imgs, msks)
    #  fn = DETR(num_queries=100)
    #  outputs = fn(inputs)
    #  pred_boxes = outputs["pred_boxes"][0]
    #  plot_boxes(boxes=pred_boxes, path="/store/plot/test_init_boxes.png")
