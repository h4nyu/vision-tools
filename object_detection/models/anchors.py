import torch
import numpy as np
import typing as t
import itertools
from typing import Any, Dict, Tuple
from torch import nn, Tensor
from object_detection.entities import (
    PascalBoxes,
    ImageBatch,
    boxmaps_to_boxes,
    BoxMaps,
    BoxMap,
    yolo_clamp,
)


class Anchors:
    def __init__(
        self,
        size: float = 1.0,
        ratios: t.List[float] = [3 / 4, 1, 4 / 3],
        scales: t.List[float] = [
            1.0,
            (1 / 2) ** (1 / 2),
            2 ** (1 / 2),
        ],
        use_cache: bool = True,
    ) -> None:
        self.use_cache = use_cache
        pairs = torch.tensor(list(itertools.product(scales, ratios)))
        self.num_anchors = len(pairs)
        self.ratios = torch.stack([pairs[:, 1], 1 / pairs[:, 1]]).t()
        self.scales = (
            pairs[:, 0]
            .view(self.num_anchors, 1)
            .expand((self.num_anchors, 2))
        ) * size
        self.cache: Dict[Tuple[int, int], PascalBoxes] = {}

    @torch.no_grad()
    def __call__(
        self, images: ImageBatch, stride: int
    ) -> PascalBoxes:
        h, w = images.shape[2:]
        device = images.device
        if self.use_cache and (h, w) in self.cache:
            return self.cache[(h, w)]
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(0, h, dtype=torch.float32, device=device)
            * stride
            + stride // 2,
            torch.arange(0, w, dtype=torch.float32, device=device)
            * stride
            + stride // 2,
        )
        box_wh = torch.tensor([stride, stride])
        box_wh = self.ratios * self.scales * box_wh
        box_wh = (
            box_wh.to(device)
            .view(self.num_anchors, 2, 1, 1)
            .expand((self.num_anchors, 2, h, w))
        )
        grid_x0y0 = (
            torch.stack([grid_x, grid_y])
            .to(device)
            .expand(self.num_anchors, 2, h, w)
            - box_wh // 2
        )

        grid_x1y1 = grid_x0y0 + box_wh
        boxes = (
            torch.cat([grid_x0y0, grid_x1y1], dim=1)
            .permute(3, 2, 0, 1)
            .contiguous()
            .view(-1, 4)
        )
        boxes = PascalBoxes(boxes)
        if self.use_cache:
            self.cache[(h, w)] = boxes
        return boxes


class EmptyAnchors:
    def __init__(
        self,
        use_cache: bool = True,
    ) -> None:
        self.use_cache = use_cache
        self.cache: Dict[Tuple[int, int], BoxMap] = {}

    def __call__(self, ref_images: Tensor) -> BoxMap:
        h, w = ref_images.shape[-2:]
        if self.use_cache:
            if (h, w) in self.cache:
                return self.cache[(h, w)]
        device = ref_images.device
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.float32) / h,
            torch.arange(w, dtype=torch.float32) / w,
        )
        box_h = torch.zeros((h, w))
        box_w = torch.zeros((h, w))
        boxmap = BoxMap(
            torch.stack([grid_x, grid_y, box_w, box_h]).to(device)
        )
        if self.use_cache:
            self.cache[(h, w)] = boxmap
        return boxmap
