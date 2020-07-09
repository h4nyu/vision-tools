import torch
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor
from typing import Tuple, List
from typing_extensions import Literal
from .centernet import CenterNet, Reg, Heatmap, GetPeaks, Sizemap, DiffMap
from .efficientdet import Anchors, RegressionModel
from .bifpn import BiFPN
from object_detection.entities import (
    PyramidIdx,
    ImageBatch,
    YoloBoxes,
    Confidences,
)
NetOutput = Tuple[Heatmap, Sizemap, DiffMap]

class ToBoxes:
    def __init__(
        self,
        threshold: float = 0.1,
        kernel_size: int = 5,
        limit: int = 100,
        count_offset: int = 1,
    ) -> None:
        self.limit = limit
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.count_offset = count_offset
        self.max_pool = partial(
            F.max_pool2d, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        )

    @torch.no_grad()
    def __call__(self, inputs: NetOutput) -> List[Tuple[YoloBoxes, Confidences]]:
        heatmap, sizemap, diffmap = inputs
        device = heatmap.device
        kpmap = (self.max_pool(heatmap) == heatmap) & (heatmap > self.threshold)
        batch_size, _, height, width = heatmap.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        rows: List[Tuple[YoloBoxes, Confidences]] = []
        for hm, km, sm, dm in zip(
            heatmap.squeeze(1), kpmap.squeeze(1), sizemap, diffmap
        ):
            kp = km.nonzero()
            confidences = hm[kp[:, 0], kp[:, 1]]
            wh = sm[:, kp[:, 0], kp[:, 1]]
            diff_wh = dm[:, kp[:, 0], kp[:, 1]].t()
            cxcy = kp[:, [1, 0]].float() / original_wh + diff_wh
            boxes = torch.cat([cxcy, wh.permute(1, 0)], dim=1)
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            rows.append(
                (YoloBoxes(boxes[sort_idx]), Confidences(confidences[sort_idx]))
            )
        return rows


class MkMaps:
    def __init__(
        self,
        sigma: float = 0.5,
        mode: Literal["length", "aspect", "constant"] = "length",
    ) -> None:
        self.sigma = sigma
        self.mode = mode

    def _mkmaps(
        self, boxes: YoloBoxes, hw: Tuple[int, int], original_hw: Tuple[int, int]
    ) -> NetOutput:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        sizemap = torch.zeros((1, 2, h, w), dtype=torch.float32).to(device)
        diffmap = torch.zeros((1, 2, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        counts = torch.tensor([box_count]).to(device)
        if box_count == 0:
            return Heatmap(heatmap), Sizemap(sizemap), DiffMap(diffmap)

        box_cxs, box_cys, _, _ = boxes.unbind(-1)
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64), torch.arange(w, dtype=torch.int64),
        )

        wh = torch.tensor([w, h]).to(device)
        cxcy = (boxes[:, :2] * wh)
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        if self.mode == "aspect":
            weight = (boxes[:, 2:] ** 2).clamp(min=1e-4).view(box_count, 2, 1, 1)
        else:
            weight = (
                (boxes[:, 2:] ** 2)
                .min(dim=1, keepdim=True)[0]
                .clamp(min=1e-4)
                .view(box_count, 1, 1, 1)
            )

        mounts = torch.exp(
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(dim=1, keepdim=True)
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        diffmap = (grid_xy - grid_cxcy) / wh.view(1, 2, 1, 1).expand_as(grid_xy)
        a, pos = ((grid_xy - grid_cxcy)**2).sum(dim=1).min(dim=0)
        print(pos.shape)
        print(diffmap.shape)
        b = diffmap.permute(3, 2, 1, 0)
        print(b.shape)
        os = b.shape
        b= b.reshape(-1, 2)
        b = b[pos.view(-1)]
        b = b.view(128, 128, 2, 1)
        b = b.permute(3, 2, 1, 0)
        print(b.shape)




        #  sizemap[:, :, cy, cx] = boxes[:, 2:].t()
        #  diffmap[:, :, cy, cx] = (boxes[:, :2] - cxcy.float() / wh).t()
        return Heatmap(heatmap), Sizemap(sizemap), DiffMap(b)


    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[YoloBoxes],
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> NetOutput:
        hms: List[Tensor] = []
        sms: List[Tensor] = []
        pms: List[Tensor] = []
        for boxes in box_batch:
            hm, sm, pm = self._mkmaps(boxes, hw, original_hw)
            hms.append(hm)
            sms.append(sm)
            pms.append(pm)

        return (
            Heatmap(torch.cat(hms, dim=0)),
            Sizemap(torch.cat(sms, dim=0)),
            DiffMap(torch.cat(pms, dim=0)),
        )
