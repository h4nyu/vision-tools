import torch
import numpy as np
import typing as t
import itertools
from typing import Any
from torch import nn, Tensor
from object_detection.entities import YoloBoxes, ImageBatch, boxmaps_to_boxes, BoxMaps


class Anchors:
    def __init__(
        self,
        ratios: t.List[float] = [0.5, 1, 2],
        scales: t.List[float] = [1.0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    ) -> None:
        pairs = torch.tensor(list(itertools.product(scales, ratios)))
        self.num_anchors = len(pairs)
        self.ratios = torch.stack([pairs[:, 1], 1 / pairs[:, 1]]).t()
        self.scales = (
            pairs[:, 0].view(self.num_anchors, 1).expand((self.num_anchors, 2))
        )

    def __call__(self, images: ImageBatch) -> YoloBoxes:
        h, w = images.shape[2:]
        device = images.device
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.float32) / h,
            torch.arange(w, dtype=torch.float32) / w,
        )
        box_wh = torch.tensor([1 / w, 1 / h])
        box_wh = self.ratios * self.scales * box_wh
        box_wh = (
            box_wh.to(device)
            .view(self.num_anchors, 2, 1, 1)
            .expand((self.num_anchors, 2, h, w))
        )
        print(box_wh.shape)
        grid_xy = (
            torch.stack([grid_x, grid_y]).to(device).expand(self.num_anchors, 2, h, w)
        )
        boxmaps = BoxMaps(torch.cat([grid_xy, box_wh], dim=1))
        return boxmaps_to_boxes(boxmaps)
