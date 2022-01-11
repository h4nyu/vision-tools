import torch, numpy as np
import itertools
from typing import Optional
from torch import nn, Tensor
from vision_tools import (
    boxmaps_to_boxes,
)
from torchvision.ops import box_convert, clip_boxes_to_image


class Anchors:
    def __init__(
        self,
        size: float = 1.0,
        ratios: list[float] = [3 / 4, 1, 4 / 3],
        scales: list[float] = [
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
            pairs[:, 0].view(self.num_anchors, 1).expand((self.num_anchors, 2))
        ) * size
        self.cache: dict[tuple[int, int], Tensor] = {}

    @torch.no_grad()
    def __call__(self, images: Tensor, stride: int) -> Tensor:
        h, w = images.shape[2:]
        device = images.device
        if self.use_cache and (h, w) in self.cache:
            return self.cache[(h, w)]
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(0, h, dtype=torch.float32, device=device) * stride
            + stride // 2,
            torch.arange(0, w, dtype=torch.float32, device=device) * stride
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
            torch.stack([grid_x, grid_y]).to(device).expand(self.num_anchors, 2, h, w)
            - box_wh // 2
        )

        grid_x1y1 = grid_x0y0 + box_wh
        boxes = (
            torch.cat([grid_x0y0, grid_x1y1], dim=1)
            .permute(2, 3, 0, 1)
            .contiguous()
            .view(-1, 4)
        )
        boxes = box_convert(
            clip_boxes_to_image(
                box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                size=(h * stride, w * stride),
            ),
            in_fmt="xyxy",
            out_fmt="cxcywh",
        )
        if self.use_cache:
            self.cache[(h, w)] = boxes
        return boxes


class EmptyAnchors:
    def __init__(
        self,
        use_cache: bool = True,
    ) -> None:
        self.use_cache = use_cache
        self.cache: dict[tuple[int, int], Tensor] = {}

    def __call__(self, ref_images: Tensor) -> Tensor:
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
        boxmap = torch.stack([grid_x, grid_y, box_w, box_h]).to(device)
        if self.use_cache:
            self.cache[(h, w)] = boxmap
        return boxmap


class Anchor:
    def __init__(
        self,
        use_cache: bool = True,
    ) -> None:
        self.use_cache = use_cache
        self.cache: dict[tuple[int, int, int], Tensor] = {}

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        stride: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        cache_key = (height, width, stride)
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        box_h = torch.ones((height, width), dtype=torch.float32)
        box_w = torch.ones((height, width), dtype=torch.float32)
        boxmap = torch.stack([grid_x, grid_y, box_w, box_h], dim=-1) * stride
        if device is not None:
            boxmap = boxmap.to(device)
        boxes = boxmap.view(-1, 4)
        boxes = box_convert(
            boxes,
            in_fmt="xywh",
            out_fmt="xyxy",
        )
        if self.use_cache:
            self.cache[cache_key] = boxes
        return boxes
