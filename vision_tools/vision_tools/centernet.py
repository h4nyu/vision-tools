import torch, math, numpy as np
from torch import Tensor
import torch.nn.functional as F
from functools import partial
from torchvision.ops import box_convert
from typing import NewType, Union, Callable, Any, List, Tuple
from torch import nn, Tensor
from logging import getLogger
from tqdm import tqdm
from vision_tools import (
    boxmap_to_boxes,
    resize_points,
)
from .mkmaps import MkMapsFn, MkBoxMapsFn
from .block import DefaultActivation, ConvBnAct, SeparableConvBnAct
from .modules import (
    FReLU,
    ConvBR2d,
    SeparableConv2d,
    SeparableConvBR2d,
    MemoryEfficientSwish,
)
from .interface import BackboneLike, FPNLike
from .assign import SimOTA
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .loss import HuberLoss, DIoULoss
from .anchors import Anchor
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import nms
from torch.utils.data import DataLoader

from pathlib import Path

logger = getLogger(__name__)


class _Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int = 1,
        depth: int = 1,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.stem = SeparableConvBnAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act=act,
        )
        self.obj_branch = nn.Sequential(
            *[
                SeparableConvBnAct(
                    in_channels=hidden_channels, out_channels=hidden_channels, act=act
                )
                for _ in range(depth)
            ],
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
            ),
        )

        self.cls_branch = nn.Sequential(
            *[
                SeparableConvBnAct(
                    in_channels=hidden_channels, out_channels=hidden_channels, act=act
                )
                for _ in range(depth)
            ],
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
            ),
        )

        self.box_branch = nn.Sequential(
            *[
                SeparableConvBnAct(
                    in_channels=hidden_channels, out_channels=hidden_channels, act=act
                )
                for _ in range(depth)
            ],
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=4,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        box_out = self.box_branch(stem)
        obj_out = self.obj_branch(stem)
        cls_out = self.cls_branch(stem)
        return torch.cat([box_out, obj_out, cls_out], dim=1)


class CenterNetHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: int,
        num_classes: int = 1,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                _Head(
                    in_channels=c,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    act=act,
                )
                for c in in_channels
            ]
        )

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        return [m(x) for m, x in zip(self.heads, feats)]


NetOutput = Tuple[Tensor, Tensor, Tensor]  # label, pos, size, count


class CenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        backbone: BackboneLike,
        neck: FPNLike,
        hidden_channels: int = 1,
        box_depth: int = 1,
        cls_depth: int = 1,
        fpn_depth: int = 1,
        out_idx: int = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.backbone = backbone
        self.neck = neck
        self.box_head = CenterNetHead(
            in_channels=neck.channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )


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
        neg_loss = neg_weight * (-(pred**alpha) * torch.log(1 - pred) * neg_mask)
        neg_loss = neg_loss.sum()
        loss = (pos_loss + neg_loss) / pos_mask.sum().clamp(min=1.0)
        return loss


# class Criterion:
#     def __init__(
#         self,
#         mk_hmmaps: MkMapsFn,
#         mk_boxmaps: MkBoxMapsFn,
#         heatmap_weight: float = 1.0,
#         box_weight: float = 1.0,
#         count_weight: float = 1.0,
#         sigma: float = 0.3,
#     ) -> None:
#         super().__init__()
#         self.hmloss = HMLoss()
#         self.boxloss = BoxLoss()
#         self.heatmap_weight = heatmap_weight
#         self.box_weight = box_weight
#         self.count_weight = count_weight
#         self.mk_hmmaps = mk_hmmaps

#     def __call__(
#         self,
#         images: Tensor,
#         netout: NetOutput,
#         gt_box_batch: List[Tensor],
#         gt_label_batch: List[Tensor],
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#         s_hm, s_bm, anchors = netout
#         _, _, orig_h, orig_w = images.shape
#         _, _, h, w = s_hm.shape
#         t_hm = self.mk_hmmaps(gt_box_batch, gt_label_batch, (h, w), (orig_h, orig_w))
#         hm_loss = self.hmloss(s_hm, t_hm) * self.heatmap_weight
#         box_loss = self.boxloss(s_bm, gt_box_batch, anchors) * self.box_weight
#         loss = hm_loss + box_loss
#         return (loss, hm_loss, box_loss, t_hm)


# class BoxLoss:
#     def __init__(
#         self,
#         use_diff: bool = True,
#     ) -> None:
#         self.loss = DIoULoss(size_average=True)
#         self.assgin = SimOTA()
#         self.use_diff = use_diff

#     def __call__(
#         self,
#         preds: Tensor,
#         gt_box_batch: List[Tensor],
#         anchormap: Tensor,
#     ) -> Tensor:
#         device = preds.device
#         _, _, h, w = preds.shape
#         box_losses: List[Tensor] = []
#         anchors = boxmap_to_boxes(anchormap)
#         for diff_map, gt_boxes in zip(preds, gt_box_batch):
#             if len(gt_boxes) == 0:
#                 continue

#             pred_boxes = boxmap_to_boxes(diff_map)
#             match_indices, positive_indices = self.assgin(anchors, gt_boxes, (w, h))
#             num_pos = positive_indices.sum()
#             if num_pos == 0:
#                 continue
#             matched_gt_boxes = gt_boxes[match_indices][positive_indices]
#             matched_pred_boxes = pred_boxes[positive_indices]
#             if self.use_diff:
#                 matched_pred_boxes = anchors[positive_indices] + matched_pred_boxes
#             box_losses.append(
#                 self.loss(
#                     box_convert(matched_pred_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
#                     box_convert(matched_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
#                 )
#             )
#         if len(box_losses) == 0:
#             return torch.tensor(0.0).to(device)
#         return torch.stack(box_losses).mean()


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
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        heatmaps, boxmaps, anchormap = inputs
        device = heatmaps.device
        kpmaps = heatmaps * (
            (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        )
        kpmaps, labelmaps = torch.max(kpmaps, dim=1)
        box_batch: List[Tensor] = []
        confidence_batch: List[Tensor] = []
        label_batch: List[Tensor] = []
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
            box_List: List[Tensor] = []
            confidence_List: List[Tensor] = []
            label_List: List[Tensor] = []

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
                box_List.append(c_boxes[nms_indices])
                confidence_List.append(c_confidences[nms_indices])
                label_List.append(c_labels[nms_indices])

            if len(confidence_List) > 0:
                confidences = torch.cat(confidence_List, dim=0)
            else:
                confidences = torch.zeros(
                    0, device=confidences.device, dtype=confidences.dtype
                )
            if len(box_List) > 0:
                boxes = torch.cat(box_List, dim=0)
            else:
                boxes = torch.zeros(0, device=boxes.device, dtype=boxes.dtype)
            if len(label_List) > 0:
                labels = torch.cat(label_List, dim=0)
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
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        device = heatmaps.device
        kpmaps = heatmaps * (
            (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        )
        kpmaps, labelmaps = torch.max(kpmaps, dim=1)
        point_batch: List[Tensor] = []
        confidence_batch: List[Tensor] = []
        label_batch: List[Tensor] = []
        _, _, hm_h, hm_w = heatmaps.shape
        for km, lm in zip(kpmaps, labelmaps):
            kp = torch.nonzero(km, as_Tuple=False)  # type: ignore
            pos_idx = (kp[:, 0], kp[:, 1])
            confidences = km[pos_idx]
            labels = lm[pos_idx]
            points: Tensor = resize_points(
                torch.stack([kp[:, 1], kp[:, 0]], dim=-1),
                scale_x=1 / hm_w,
                scale_y=1 / hm_h,
            )
            unique_labels = labels.unique()
            point_List: List[Tensor] = []
            confidence_List: List[Tensor] = []
            label_List: List[Tensor] = []

            for c in unique_labels:
                cls_indices = labels == c
                if cls_indices.sum() == 0:
                    continue
                c_points = points[cls_indices]
                c_confidences = confidences[cls_indices]
                c_labels = labels[cls_indices]
                c_sort_indices = c_confidences.argsort(descending=True)
                point_List.append(c_points[c_sort_indices])
                confidence_List.append(c_confidences[c_sort_indices])
                label_List.append(c_labels[c_sort_indices])

            if len(confidence_List) > 0:
                confidences = torch.cat(confidence_List, dim=0)
            else:
                confidences = torch.zeros(
                    0, device=confidences.device, dtype=confidences.dtype
                )
            if len(point_List) > 0:
                points = torch.cat(point_List, dim=0)
            else:
                points = torch.zeros(0, device=points.device, dtype=points.dtype)
            if len(label_List) > 0:
                labels = torch.cat(label_List, dim=0)
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
