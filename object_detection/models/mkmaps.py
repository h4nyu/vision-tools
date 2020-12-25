from typing import Tuple, List, Callable, NewType
from typing_extensions import Literal
import torch
from object_detection.entities import YoloBoxes, BoxMaps

Heatmaps = NewType("Heatmaps", torch.Tensor)  # [B, C, H, W]
MkMapsFn = Callable[
    [
        List[YoloBoxes],
        Tuple[int, int],
        Tuple[int, int],
    ],
    Heatmaps,
]

MkBoxMapsFn = Callable[[List[YoloBoxes], Heatmaps], BoxMaps]


class MkMapsBase:
    def _mkmaps(
        self,
        boxes: YoloBoxes,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        ...

    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[YoloBoxes],
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        hms: List[torch.Tensor] = []
        for boxes in box_batch:
            hm = self._mkmaps(boxes, hw, original_hw)
            hms.append(hm)
        return Heatmaps(torch.cat(hms, dim=0))


class MkCrossMaps(MkMapsBase):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def _mkmaps(
        self,
        boxes: YoloBoxes,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(
            device
        )
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)
        step_w = 1.0 / w
        step_h = 1.0 / h
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(
                start=0.0,
                end=1.0,
                step=step_h,
                dtype=torch.float32,
            ).to(device),
            torch.arange(
                start=0.0,
                end=1.0,
                step=step_w,
                dtype=torch.float32,
            ).to(device),
        )
        grid_cxcy = (
            (boxes[:, :2])
            .view(box_count, 2, 1, 1)
            .expand((box_count, 2, h, w))
        )
        grid_wh = (
            (boxes[:, 2:])
            .view(box_count, 2, 1, 1)
            .expand((box_count, 2, h, w))
        )
        grid_xy = torch.stack([grid_x, grid_y]).expand(
            (box_count, 2, h, w)
        )
        grid_xycxcywh = torch.cat(
            [grid_cxcy, grid_xy, grid_wh], dim=1
        )
        mounts = (
            (grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]).abs()
            < step_w
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h
            )
        )
        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                - grid_xycxcywh[:, 4, ::] * self.alpha / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w / 2
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h / 2
            )
        )
        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                + grid_xycxcywh[:, 4, ::] * self.alpha / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w / 2
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h / 2
            )
        )
        mounts |= (
            (
                grid_xycxcywh[:, 3, ::]
                - grid_xycxcywh[:, 5, ::] * self.alpha / 2
                - grid_xycxcywh[:, 1, ::]
            ).abs()
            < step_w / 2
        ) & (
            (
                (
                    grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]
                ).abs()
                < step_w / 2
            )
        )
        mounts |= (
            (
                grid_xycxcywh[:, 3, ::]
                + grid_xycxcywh[:, 5, ::] * self.alpha / 2
                - grid_xycxcywh[:, 1, ::]
            ).abs()
            < step_w / 2
        ) & (
            (
                (
                    grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]
                ).abs()
                < step_w / 2
            )
        )
        mounts = mounts.view(box_count, 1, h, w).float()
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


class MkCornerMaps(MkMapsBase):
    def __init__(self) -> None:
        ...

    def _mkmaps(
        self,
        boxes: YoloBoxes,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(
            device
        )
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)
        step_w = 1.0 / w
        step_h = 1.0 / h
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(
                start=0.0,
                end=1.0,
                step=step_h,
                dtype=torch.float32,
            ).to(device),
            torch.arange(
                start=0.0,
                end=1.0,
                step=step_w,
                dtype=torch.float32,
            ).to(device),
        )
        grid_cxcy = (
            (boxes[:, :2])
            .view(box_count, 2, 1, 1)
            .expand((box_count, 2, h, w))
        )
        grid_wh = (
            (boxes[:, 2:])
            .view(box_count, 2, 1, 1)
            .expand((box_count, 2, h, w))
        )
        grid_xy = torch.stack([grid_x, grid_y]).expand(
            (box_count, 2, h, w)
        )
        grid_xycxcywh = torch.cat(
            [grid_cxcy, grid_xy, grid_wh], dim=1
        )
        mounts = (
            (
                grid_xycxcywh[:, 2, ::]
                - grid_xycxcywh[:, 4, ::] / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::]
                    - grid_xycxcywh[:, 5, ::] / 2
                    - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h
            )
        )
        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                + grid_xycxcywh[:, 4, ::] / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::]
                    + grid_xycxcywh[:, 5, ::] / 2
                    - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h
            )
        )

        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                - grid_xycxcywh[:, 4, ::] / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::]
                    + grid_xycxcywh[:, 5, ::] / 2
                    - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h
            )
        )

        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                + grid_xycxcywh[:, 4, ::] / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w
        ) & (
            (
                (
                    grid_xycxcywh[:, 3, ::]
                    - grid_xycxcywh[:, 5, ::] / 2
                    - grid_xycxcywh[:, 1, ::]
                ).abs()
                < step_h
            )
        )
        mounts = mounts.view(box_count, 1, h, w).float()
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


GaussianMapMode = Literal["length", "aspect", "constant"]


class MkGaussianMaps(MkMapsBase):
    def __init__(
        self,
        sigma: float = 0.5,
        mode: GaussianMapMode = "length",
    ) -> None:
        self.sigma = sigma
        self.mode = mode

    def _mkmaps(
        self,
        boxes: YoloBoxes,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(
            device
        )
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = (
            torch.stack([grid_x, grid_y])
            .to(device)
            .expand((box_count, 2, h, w))
        )
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        if self.mode == "aspect":
            weight = (
                (boxes[:, 2:] ** 2)
                .clamp(min=1e-4)
                .view(box_count, 2, 1, 1)
            )
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
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(
                dim=1, keepdim=True
            )
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


class MkFillMaps(MkMapsBase):
    def __init__(
        self, sigma: float = 0.5, use_overlap: bool = False
    ) -> None:
        self.sigma = sigma
        self.use_overlap = use_overlap

    def _mkmaps(
        self,
        boxes: YoloBoxes,
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(
            device
        )
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = (
            torch.stack([grid_x, grid_y])
            .to(device)
            .expand((box_count, 2, h, w))
        )
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        grid_wh = img_wh.view(1, 2, 1, 1).expand_as(grid_xy)
        min_wh = (1.0 / img_wh.float()).view(1, 2).expand_as(box_wh)
        clamped_wh = torch.max(box_wh * 0.5 * self.sigma, min_wh)
        mounts = (
            (grid_xy.float() / grid_wh)
            - (grid_cxcy.float() / grid_wh)
        ).abs() < clamped_wh.view(box_count, 2, 1, 1).expand_as(
            grid_xy
        )
        mounts, _ = mounts.min(dim=1, keepdim=True)
        mounts = mounts.float()
        if self.use_overlap:
            heatmap, _ = mounts.max(dim=0, keepdim=True)
        else:
            heatmap = mounts.sum(dim=0, keepdim=True)
            heatmap = heatmap.eq(1.0).float()

        return Heatmaps(heatmap)


class MkCenterBoxMaps:
    def __init__(
        self,
    ) -> None:
        ...

    def _mkmaps(
        self, boxes: YoloBoxes, heatmaps: Heatmaps
    ) -> BoxMaps:
        device = boxes.device
        _, _, h, w = heatmaps.shape
        boxmaps = torch.zeros((1, 4, h, w), dtype=torch.float32).to(
            device
        )
        box_count = len(boxes)
        if box_count == 0:
            return BoxMaps(boxmaps)
        wh = torch.tensor([w, h]).to(device)
        cxcy = (boxes[:, :2] * wh).long()
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        boxmaps[:, :, cy, cx] = boxes.t()
        return BoxMaps(boxmaps)

    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[YoloBoxes],
        heatmaps: Heatmaps,
    ) -> BoxMaps:
        bms: List[torch.Tensor] = []
        for boxes in box_batch:
            bms.append(self._mkmaps(boxes, heatmaps))

        return BoxMaps(torch.cat(bms, dim=0))
