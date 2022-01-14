from torch import Tensor
from typing import Callable, List, Tuple
from typing_extensions import Literal
import torch
from vision_tools import Number, resize_points

MkMapsFn = Callable[
    [
        List[Tensor],
        List[Tensor],
        Tuple[int, int],
        Tuple[int, int],
    ],
    Tensor,
]

MkBoxMapsFn = Callable[[List[Tensor], Tensor], Tensor]


class MkMapsBase:
    num_classes: int

    def _mkmaps(
        self,
        boxes: Tensor,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Tensor:
        ...

    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[Tensor],
        label_batch: List[Tensor],
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Tensor:
        hms: List[Tensor] = []
        for boxes, labels in zip(box_batch, label_batch):
            hm = torch.cat(
                [
                    self._mkmaps(boxes[labels == i], hw, original_hw)
                    for i in range(self.num_classes)
                ],
                dim=1,
            )
            hms.append(hm)
        return torch.cat(hms, dim=0)


GaussianMapMode = Literal["length", "aspect", "constant"]


class MkGaussianMaps(MkMapsBase):
    def __init__(
        self,
        num_classes: int,
        sigma: float = 0.5,
        mode: GaussianMapMode = "length",
    ) -> None:
        self.sigma = sigma
        self.mode = mode
        self.num_classes = num_classes

    def _mkmaps(
        self,
        boxes: Tensor,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Tensor:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return heatmap

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        if self.mode == "aspect":
            weight = (boxes[:, 2:] ** 2).clamp(min=1e-4).view(box_count, 2, 1, 1)
        elif self.mode == "length":
            weight = (
                (boxes[:, 2:] ** 2)
                .min(dim=1, keepdim=True)[0]
                .clamp(min=1e-4)
                .view(box_count, 1, 1, 1)
            )
        else:
            weight = torch.ones((box_count, 1, 1, 1)).to(device)
        mounts = torch.exp(
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(dim=1, keepdim=True)
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return heatmap


class MkFillMaps(MkMapsBase):
    def __init__(self, sigma: float = 0.5, use_overlap: bool = False) -> None:
        self.sigma = sigma
        self.use_overlap = use_overlap

    def _mkmaps(
        self,
        boxes: Tensor,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Tensor:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return heatmap

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        grid_wh = img_wh.view(1, 2, 1, 1).expand_as(grid_xy)
        min_wh = (1.0 / img_wh.float()).view(1, 2).expand_as(box_wh)
        clamped_wh = torch.max(box_wh * 0.5 * self.sigma, min_wh)
        mounts = (
            (grid_xy.float() / grid_wh) - (grid_cxcy.float() / grid_wh)
        ).abs() < clamped_wh.view(box_count, 2, 1, 1).expand_as(grid_xy)
        mounts, _ = mounts.min(dim=1, keepdim=True)
        mounts = mounts.float()
        if self.use_overlap:
            heatmap, _ = mounts.max(dim=0, keepdim=True)
        else:
            heatmap = mounts.sum(dim=0, keepdim=True)
            heatmap = heatmap.eq(1.0).float()

        return heatmap


class MkCenterBoxMaps:
    def __init__(
        self,
    ) -> None:
        ...

    def _mkmaps(self, boxes: Tensor, heatmaps: Tensor) -> Tensor:
        device = boxes.device
        _, _, h, w = heatmaps.shape
        boxmaps = torch.zeros((1, 4, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return boxmaps
        wh = torch.tensor([w, h]).to(device)
        cxcy = (boxes[:, :2] * wh).long()
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        boxmaps[:, :, cy, cx] = boxes.t()
        return boxmaps

    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[Tensor],
        heatmaps: Tensor,
    ) -> Tensor:
        bms: List[Tensor] = []
        for boxes in box_batch:
            bms.append(self._mkmaps(boxes, heatmaps))

        return torch.cat(bms, dim=0)


class MkPointMaps:
    num_classes: int

    def __init__(
        self,
        num_classes: int,
        sigma: float = 0.3,
    ) -> None:
        self.num_classes = num_classes
        self.sigma = sigma

    def _mkmaps(
        self,
        points: Tensor,
        h: int,
        w: int,
    ) -> Tensor:
        device = points.device
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        count = len(points)
        if count == 0:
            return heatmap

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((count, 2, h, w))
        grid_cxcy = points.view(count, 2, 1, 1).expand_as(grid_xy)
        weight = torch.ones((count, 1, 1, 1)).to(device)
        mounts = torch.exp(
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(dim=1, keepdim=True)
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return heatmap

    @torch.no_grad()
    def __call__(
        self,
        point_batch: List[Tensor],
        label_batch: List[Tensor],
        h: int,
        w: int,
    ) -> Tensor:
        hms: List[Tensor] = []
        for points, labels in zip(point_batch, label_batch):
            points = resize_points(points, scale_x=w, scale_y=h)
            hm = torch.cat(
                [
                    self._mkmaps(points[labels == i], h=h, w=w)
                    for i in range(self.num_classes)
                ],
                dim=1,
            )
            hms.append(hm)
        return torch.cat(hms, dim=0)
