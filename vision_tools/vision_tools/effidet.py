import math
from functools import partial
from itertools import product as product
from logging import getLogger
from pathlib import Path
from typing import *
from typing import Any, Callable, List, NewType, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torchvision.ops.boxes import box_iou
from tqdm import tqdm

from vision_tools import filter_limit

from .anchors import Anchors
from .atss import ATSS
from .bifpn import FP, BiFPN
from .bottlenecks import SENextBottleneck2d
from .interface import BackboneLike
from .loss import DIoU, DIoULoss, FocalLoss, HuberLoss
from .modules import (
    ConvBR2d,
    MemoryEfficientSwish,
    Mish,
    SeparableConv2d,
    SeparableConvBR2d,
)

logger = getLogger(__name__)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        num_anchors: int = 9,
        num_classes: int = 80,
    ) -> None:
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    SeparableConvBR2d(in_channels, in_channels),
                    MemoryEfficientSwish(),
                )
                for _ in range(depth)
            ]
        )
        self.out = SeparableConv2d(
            in_channels,
            num_anchors * num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, width, height, channels = x.shape
        x = x.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return x.contiguous().view(batch_size, -1, self.num_classes).sigmoid()


class RegressionModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    SeparableConvBR2d(in_channels, in_channels),
                    MemoryEfficientSwish(),
                )
                for _ in range(depth)
            ]
        )
        self.out = SeparableConv2d(
            in_channels,
            num_anchors * 4,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, 4)
        return x


NetOutput = Tuple[List[Tensor], List[Tensor], List[Tensor]]


def _init_weight(m: nn.Module) -> None:
    """Weight initialization as per Tensorflow official implementations."""
    if isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: BackboneLike,
        channels: int = 64,
        out_ids: List[int] = [6, 7],
        anchors: Anchors = Anchors(),
        fpn_depth: int = 1,
        box_depth: int = 1,
        cls_depth: int = 1,
    ) -> None:
        super().__init__()
        self.out_ids = np.array(out_ids) - 3
        self.anchors = anchors
        self.backbone = backbone
        self.neck = nn.Sequential(*[BiFPN(channels=channels) for _ in range(fpn_depth)])
        self.box_reg = RegressionModel(
            depth=box_depth,
            in_channels=channels,
            num_anchors=self.anchors.num_anchors,
        )
        self.classification = ClassificationModel(
            channels,
            depth=cls_depth,
            num_classes=num_classes,
            num_anchors=self.anchors.num_anchors,
        )
        for n, m in self.named_modules():
            if "backbone" not in n:
                _init_weight(m)

    def forward(self, images: Tensor) -> NetOutput:
        features = self.backbone(images)
        features = self.neck(features)
        anchor_levels = [self.anchors(features[i], 2 ** (i + 1)) for i in self.out_ids]
        box_levels = [self.box_reg(features[i]) for i in self.out_ids]
        label_levels = [self.classification(features[i]) for i in self.out_ids]
        return (
            anchor_levels,
            box_levels,
            label_levels,
        )


class Criterion:
    def __init__(
        self,
        num_classes: int = 1,
        topk: int = 13,
        box_weight: float = 3.0,
        cls_weight: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.box_loss = DIoULoss(size_average=True)
        self.atss = ATSS(topk)
        self.cls_loss = FocalLoss()

    def __call__(
        self,
        images: Tensor,
        net_output: NetOutput,
        gt_boxes_List: List[Tensor],
        gt_classes_List: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (
            anchor_levels,
            box_reg_levels,
            cls_pred_levels,
        ) = net_output
        device = anchor_levels[0].device
        batch_size = box_reg_levels[0].shape[0]
        _, _, h, w = images.shape
        anchors = torch.cat([*anchor_levels], dim=0)
        box_preds = torch.cat([*box_reg_levels], dim=1)
        cls_preds = torch.cat([*cls_pred_levels], dim=1)

        box_losses = torch.zeros(batch_size, device=device)
        cls_losses = torch.zeros(batch_size, device=device)
        for batch_id, (gt_boxes, gt_lables, box_pred, cls_pred,) in enumerate(
            zip(
                gt_boxes_List,
                gt_classes_List,
                box_preds,
                cls_preds,
            )
        ):
            pos_ids = self.atss(
                anchors,
                gt_boxes,
            )
            matched_gt_boxes = gt_boxes[pos_ids[:, 0]]
            matched_pred_boxes = anchors[pos_ids[:, 1]] + box_pred[pos_ids[:, 1]]
            cls_target = torch.zeros(
                cls_pred.shape, device=device, dtype=cls_pred.dtype
            )
            cls_target[pos_ids[:, 1], gt_lables[pos_ids[:, 0]].long()] = 1.0

            num_pos = max(len(pos_ids), 1)
            cls_losses[batch_id] = (
                self.cls_loss(
                    cls_pred,
                    cls_target,
                ).sum()
                / num_pos
            )

            if len(gt_boxes) == 0:
                continue
            box_losses[batch_id] = self.box_loss(
                matched_gt_boxes,
                matched_pred_boxes,
            ).mean()

        box_loss = box_losses.mean() * self.box_weight
        cls_loss = cls_losses.mean() * self.cls_weight
        loss = box_loss + cls_loss
        return loss, box_loss, cls_loss


class PreProcess:
    def __init__(self, device: Any, non_blocking: bool = True) -> None:
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking

    def __call__(
        self,
        batch: Tuple[Tensor, List[Tensor], List[Tensor]],
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        image_batch, boxes_batch, label_batch = batch
        return (
            image_batch.to(
                self.device,
                non_blocking=self.non_blocking,
            ),
            [
                x.to(
                    self.device,
                    non_blocking=self.non_blocking,
                )
                for x in boxes_batch
            ],
            [
                x.to(
                    self.device,
                    non_blocking=self.non_blocking,
                )
                for x in label_batch
            ],
        )


class ToBoxes:
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def __call__(
        self, net_output: NetOutput
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        (
            anchor_levels,
            box_diff_levels,
            labels_levels,
        ) = net_output
        box_batch = []
        confidence_batch = []
        label_batch = []
        anchors = torch.cat(anchor_levels, dim=0)  # type: ignore
        box_diffs = torch.cat(box_diff_levels, dim=1)  # type:ignore
        labels_batch = torch.cat(labels_levels, dim=1)  # type:ignore
        for box_diff, preds in zip(box_diffs, labels_batch):
            boxes = anchors + box_diff
            confidences, labels = preds.max(dim=1)
            filter_idx = confidences > self.confidence_threshold
            confidences = confidences[filter_idx]
            labels = labels[filter_idx]
            boxes = boxes[filter_idx]
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

                sort_indices = c_confidences.argsort(descending=True)
                c_boxes = c_boxes[sort_indices]
                c_confidences = c_confidences[sort_indices]
                c_labels = c_labels[sort_indices]

                nms_indices = nms(
                    c_boxes,
                    c_confidences,
                    self.iou_threshold,
                )
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
