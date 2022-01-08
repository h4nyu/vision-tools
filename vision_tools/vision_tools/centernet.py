import torch, math, numpy as np
from typing import Literal
from torch import Tensor
import torch.nn.functional as F
from functools import partial
from torchvision.ops import box_convert
from typing import (
    NewType,
    Union,
    Callable,
    Any,
)
from torch import nn, Tensor
from logging import getLogger
from tqdm import tqdm
from vision_tools import (
    boxmap_to_boxes,
    resize_points,
)
from .mkmaps import MkMapsFn, MkBoxMapsFn
from .modules import (
    FReLU,
    ConvBR2d,
    SeparableConv2d,
    SeparableConvBR2d,
    MemoryEfficientSwish,
)
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .loss import HuberLoss, DIoULoss
from .anchors import EmptyAnchors
from .matcher import NearnestMatcher, CenterMatcher
from vision_tools.meters import MeanMeter
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import nms
from torch.utils.data import DataLoader
from vision_tools.model_loader import ModelLoader

from pathlib import Path

logger = getLogger(__name__)


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    SeparableConvBR2d(in_channels, in_channels),
                    MemoryEfficientSwish(),
                )
                for _ in range(depth)
            ]
        )

        self.out = nn.Sequential(
            SeparableConv2d(
                in_channels,
                out_channels,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.out(x)
        return x


NetOutput = tuple[Tensor, Tensor, Tensor]  # label, pos, size, count


class CenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        backbone: nn.Module,
        box_depth: int = 1,
        cls_depth: int = 1,
        fpn_depth: int = 1,
        out_idx: int = 4,
    ) -> None:
        super().__init__()
        self.out_idx = out_idx - 3
        self.channels = channels
        self.backbone = backbone
        self.fpn = nn.Sequential(*[BiFPN(channels=channels) for _ in range(fpn_depth)])
        self.hm_reg = nn.Sequential(
            Head(
                in_channels=channels,
                out_channels=num_classes,
                depth=cls_depth,
            ),
            nn.Sigmoid(),
        )
        self.box_reg = nn.Sequential(
            Head(
                in_channels=channels,
                out_channels=4,
                depth=box_depth,
            )
        )
        self.anchors = EmptyAnchors()

    def forward(self, x: Tensor) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmaps = self.hm_reg(fp[self.out_idx])
        anchors = self.anchors(heatmaps)
        boxmaps = self.box_reg(fp[self.out_idx])
        return (heatmaps, boxmaps, anchors)


class HMLoss(nn.Module):
    """
    Modified focal loss
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 2.0,
        eps: float = 5e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        pred: 0-1 [B, C,..]
        gt: 0-1 [B, C,..]
        """
        alpha = self.alpha
        beta = self.beta
        eps = self.eps
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        pos_loss = pos_loss.sum()

        neg_weight = (1 - gt) ** beta
        neg_loss = neg_weight * (-(pred ** alpha) * torch.log(1 - pred) * neg_mask)
        neg_loss = neg_loss.sum()
        loss = (pos_loss + neg_loss) / pos_mask.sum().clamp(min=1.0)
        return loss


class Criterion:
    def __init__(
        self,
        mk_hmmaps: MkMapsFn,
        mk_boxmaps: MkBoxMapsFn,
        heatmap_weight: float = 1.0,
        box_weight: float = 1.0,
        count_weight: float = 1.0,
        sigma: float = 0.3,
    ) -> None:
        super().__init__()
        self.hmloss = HMLoss()
        self.boxloss = BoxLoss()
        self.heatmap_weight = heatmap_weight
        self.box_weight = box_weight
        self.count_weight = count_weight
        self.mk_hmmaps = mk_hmmaps

    def __call__(
        self,
        images: Tensor,
        netout: NetOutput,
        gt_box_batch: list[Tensor],
        gt_label_batch: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s_hm, s_bm, anchors = netout
        _, _, orig_h, orig_w = images.shape
        _, _, h, w = s_hm.shape
        t_hm = self.mk_hmmaps(gt_box_batch, gt_label_batch, (h, w), (orig_h, orig_w))
        hm_loss = self.hmloss(s_hm, t_hm) * self.heatmap_weight
        box_loss = self.boxloss(s_bm, gt_box_batch, anchors) * self.box_weight
        loss = hm_loss + box_loss
        return (loss, hm_loss, box_loss, t_hm)


class BoxLoss:
    def __init__(
        self,
        matcher: Any = NearnestMatcher(),
        use_diff: bool = True,
    ) -> None:
        self.matcher = matcher
        self.loss = DIoULoss(size_average=True)
        self.use_diff = use_diff

    def __call__(
        self,
        preds: Tensor,
        gt_box_batch: list[Tensor],
        anchormap: Tensor,
    ) -> Tensor:
        device = preds.device
        _, _, h, w = preds.shape
        box_losses: list[Tensor] = []
        anchors = boxmap_to_boxes(anchormap)
        for diff_map, gt_boxes in zip(preds, gt_box_batch):
            if len(gt_boxes) == 0:
                continue

            pred_boxes = boxmap_to_boxes(diff_map)
            match_indices, positive_indices = self.matcher(anchors, gt_boxes, (w, h))
            num_pos = positive_indices.sum()
            if num_pos == 0:
                continue
            matched_gt_boxes = gt_boxes[match_indices][positive_indices]
            matched_pred_boxes = pred_boxes[positive_indices]
            if self.use_diff:
                matched_pred_boxes = anchors[positive_indices] + matched_pred_boxes
            box_losses.append(
                self.loss(
                    box_convert(matched_pred_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                    box_convert(matched_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                )
            )
        if len(box_losses) == 0:
            return torch.tensor(0.0).to(device)
        return torch.stack(box_losses).mean()


class ToBoxes:
    def __init__(
        self,
        threshold: float = 0.1,
        iou_threshold: float = 0.5,
        kernel_size: int = 3,
        limit: int = 100,
        use_diff: bool = True,
    ) -> None:
        self.limit = limit
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.iou_threshold = iou_threshold
        self.use_diff = use_diff
        self.max_pool = partial(
            F.max_pool2d,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

    @torch.no_grad()
    def __call__(
        self, inputs: NetOutput
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        heatmaps, boxmaps, anchormap = inputs
        device = heatmaps.device
        kpmaps = heatmaps * (
            (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        )
        kpmaps, labelmaps = torch.max(kpmaps, dim=1)
        box_batch: list[Tensor] = []
        confidence_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        for km, lm, bm in zip(kpmaps, labelmaps, boxmaps):
            kp = torch.nonzero(km, as_tuple=False)  # type: ignore
            pos_idx = (kp[:, 0], kp[:, 1])
            confidences = km[pos_idx]
            labels = lm[pos_idx]
            if self.use_diff:
                boxes = (
                    anchormap[:, pos_idx[0], pos_idx[1]].t()
                    + bm[:, pos_idx[0], pos_idx[1]].t()
                )
            else:
                boxes = bm[:, pos_idx[0], pos_idx[1]].t()

            unique_labels = labels.unique()
            box_list: list[Tensor] = []
            confidence_list: list[Tensor] = []
            label_list: list[Tensor] = []

            for c in unique_labels:
                cls_indices = labels == c
                if cls_indices.sum() == 0:
                    continue

                c_boxes = boxes[cls_indices]
                c_confidences = confidences[cls_indices]
                c_labels = labels[cls_indices]
                nms_indices = nms(
                    box_convert(c_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                    c_confidences,
                    self.iou_threshold,
                )[: self.limit]
                box_list.append(c_boxes[nms_indices])
                confidence_list.append(c_confidences[nms_indices])
                label_list.append(c_labels[nms_indices])

            if len(confidence_list) > 0:
                confidences = torch.cat(confidence_list, dim=0)
            else:
                confidences = torch.zeros(
                    0, device=confidences.device, dtype=confidences.dtype
                )
            if len(box_list) > 0:
                boxes = torch.cat(box_list, dim=0)
            else:
                boxes = torch.zeros(0, device=boxes.device, dtype=boxes.dtype)
            if len(label_list) > 0:
                labels = torch.cat(label_list, dim=0)
            else:
                labels = torch.zeros(0, device=labels.device, dtype=labels.dtype)

            sort_indices = confidences.argsort(descending=True)
            boxes = boxes[sort_indices]
            confidences = confidences[sort_indices]
            labels = labels[sort_indices]
            box_batch.append(boxes)
            confidence_batch.append(confidences)
            label_batch.append(labels)
        return box_batch, confidence_batch, label_batch


class ToPoints:
    def __init__(
        self,
        threshold: float = 0.1,
        iou_threshold: float = 0.5,
        kernel_size: int = 3,
        limit: int = 100,
        use_diff: bool = True,
    ) -> None:
        self.limit = limit
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.iou_threshold = iou_threshold
        self.use_diff = use_diff
        self.max_pool = partial(
            F.max_pool2d,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

    @torch.no_grad()
    def __call__(
        self,
        heatmaps: Tensor,
        w: int,
        h: int,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        device = heatmaps.device
        kpmaps = heatmaps * (
            (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        )
        kpmaps, labelmaps = torch.max(kpmaps, dim=1)
        point_batch: list[Tensor] = []
        confidence_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        _, _, hm_h, hm_w = heatmaps.shape
        for km, lm in zip(kpmaps, labelmaps):
            kp = torch.nonzero(km, as_tuple=False)  # type: ignore
            pos_idx = (kp[:, 0], kp[:, 1])
            confidences = km[pos_idx]
            labels = lm[pos_idx]
            points: Tensor = resize_points(
                torch.stack([kp[:, 1], kp[:, 0]], dim=-1),
                scale_x=1 / hm_w,
                scale_y=1 / hm_h,
            )
            unique_labels = labels.unique()
            point_list: list[Tensor] = []
            confidence_list: list[Tensor] = []
            label_list: list[Tensor] = []

            for c in unique_labels:
                cls_indices = labels == c
                if cls_indices.sum() == 0:
                    continue
                c_points = points[cls_indices]
                c_confidences = confidences[cls_indices]
                c_labels = labels[cls_indices]
                c_sort_indices = c_confidences.argsort(descending=True)
                point_list.append(c_points[c_sort_indices])
                confidence_list.append(c_confidences[c_sort_indices])
                label_list.append(c_labels[c_sort_indices])

            if len(confidence_list) > 0:
                confidences = torch.cat(confidence_list, dim=0)
            else:
                confidences = torch.zeros(
                    0, device=confidences.device, dtype=confidences.dtype
                )
            if len(point_list) > 0:
                points = torch.cat(point_list, dim=0)
            else:
                points = torch.zeros(0, device=points.device, dtype=points.dtype)
            if len(label_list) > 0:
                labels = torch.cat(label_list, dim=0)
            else:
                labels = torch.zeros(0, device=labels.device, dtype=labels.dtype)

            sort_indices = confidences.argsort(descending=True)
            points = points[sort_indices]
            confidences = confidences[sort_indices]
            labels = labels[sort_indices]
            point_batch.append(points)
            confidence_batch.append(confidences)
            label_batch.append(labels)
        return point_batch, confidence_batch, label_batch
