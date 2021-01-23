import numpy as np
import typing as t
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from object_detection.entities import (
    Image,
    ImageId,
    Confidences,
    PredictionSample,
    Labels,
    PascalBoxes,
    yolo_to_pascal,
    ImageBatch,
    yolo_clamp,
)
from object_detection.model_loader import ModelLoader
from object_detection.meters import MeanMeter
from object_detection.utils import DetectionPlot
from typing import Any, List, Tuple, NewType, Callable
from torchvision.ops.boxes import box_iou
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torch import nn, Tensor
from itertools import product as product
from logging import getLogger
from pathlib import Path
from typing_extensions import Literal
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import DIoU, HuberLoss, DIoULoss, FocalLoss
from .modules import ConvBR2d, SeparableConv2d, SeparableConvBR2d, Mish
from .atss import ATSS
from .anchors import Anchors
from .tta import VFlipTTA, HFlipTTA

logger = getLogger(__name__)

TrainSample = Tuple[ImageId, Image, PascalBoxes, Labels]


def collate_fn(
    batch: List[TrainSample],
) -> Tuple[ImageBatch, List[PascalBoxes], List[Labels], List[ImageId]]:
    images: List[t.Any] = []
    id_batch: List[ImageId] = []
    box_batch: List[PascalBoxes] = []
    label_batch: List[Labels] = []
    for id, img, boxes, labels in batch:
        c, h, w = img.shape
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return (
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
        id_batch,
    )


def prediction_collate_fn(
    batch: List[PredictionSample],
) -> Tuple[ImageBatch, List[ImageId]]:
    images: List[Any] = []
    id_batch: List[ImageId] = []
    for id, img in batch:
        images.append(img)
        id_batch.append(id)
    return ImageBatch(torch.stack(images)), id_batch


class Visualize:
    def __init__(
        self,
        out_dir: str,
        prefix: str,
        limit: int = 1,
        use_alpha: bool = True,
        show_confidences: bool = True,
        transforms: Any = None,
    ) -> None:
        self.prefix = prefix
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.use_alpha = use_alpha
        self.show_confidences = show_confidences
        self.transforms = transforms

    def __call__(
        self,
        image_batch: ImageBatch,
        src: Tuple[List[PascalBoxes], List[Confidences], List[Labels]],
        tgt: Tuple[List[PascalBoxes], List[Labels]],
    ) -> None:
        image_batch = ImageBatch(image_batch[: self.limit])
        gt_boxes, gt_labels = tgt
        gt_boxes = gt_boxes[: self.limit]
        gt_labels = gt_labels[: self.limit]
        box_batch, confidence_batch, label_batch = src
        _, _, h, w = image_batch.shape
        for i, (img, boxes, confidences, labels, gtb, gtl,) in enumerate(
            zip(
                image_batch,
                box_batch,
                confidence_batch,
                label_batch,
                gt_boxes,
                gt_labels,
            )
        ):
            plot = DetectionPlot(
                self.transforms(img) if self.transforms is not None else img
            )
            plot.draw_boxes(boxes=gtb, color="blue", labels=gtl)
            plot.draw_boxes(
                boxes=boxes, color="red", labels=labels, confidences=confidences
            )
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")


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
            *[SENextBottleneck2d(in_channels, in_channels) for _ in range(depth)]
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
            *[SENextBottleneck2d(in_channels, in_channels) for _ in range(depth)]
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


BoxDiff = NewType("BoxDiff", Tensor)
BoxDiffs = NewType("BoxDiffs", Tensor)

NetOutput = Tuple[List[PascalBoxes], List[BoxDiffs], List[Tensor]]


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
        backbone: nn.Module,
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

    def forward(self, images: ImageBatch) -> NetOutput:
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
        images: ImageBatch,
        net_output: NetOutput,
        gt_boxes_list: List[PascalBoxes],
        gt_classes_list: List[Labels],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (
            anchor_levels,
            box_reg_levels,
            cls_pred_levels,
        ) = net_output
        device = anchor_levels[0].device
        batch_size = box_reg_levels[0].shape[0]
        _, _, h, w = images.shape
        anchors = PascalBoxes(torch.cat([*anchor_levels], dim=0))
        box_preds = torch.cat([*box_reg_levels], dim=1)
        cls_preds = torch.cat([*cls_pred_levels], dim=1)

        box_losses = torch.zeros(batch_size, device=device)
        cls_losses = torch.zeros(batch_size, device=device)
        for batch_id, (gt_boxes, gt_lables, box_pred, cls_pred,) in enumerate(
            zip(
                gt_boxes_list,
                gt_classes_list,
                box_preds,
                cls_preds,
            )
        ):
            pos_ids = self.atss(
                anchors,
                PascalBoxes(gt_boxes),
            )
            matched_gt_boxes = gt_boxes[pos_ids[:, 0]]
            matched_pred_boxes = anchors[pos_ids[:, 1]] + box_pred[pos_ids[:, 1]]
            cls_target = torch.zeros(cls_pred.shape, device=device)
            cls_target[pos_ids[:, 1], gt_lables[pos_ids[:, 0]].long()] = 1
            cls_losses[batch_id] = self.cls_loss(
                cls_pred.float(),
                cls_target.float(),
            ).sum()
            box_losses[batch_id] = self.box_loss(
                PascalBoxes(matched_gt_boxes),
                PascalBoxes(matched_pred_boxes),
            ).mean()

        box_loss = box_losses.mean() * self.box_weight
        cls_loss = cls_losses.mean() * self.cls_weight
        loss = box_loss + cls_loss
        return loss, box_loss, cls_loss


class PreProcess:
    def __init__(self, device: t.Any, non_blocking: bool = True) -> None:
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking

    def __call__(
        self,
        batch: t.Tuple[ImageBatch, List[PascalBoxes], List[Labels]],
    ) -> t.Tuple[ImageBatch, List[PascalBoxes], List[Labels]]:
        image_batch, boxes_batch, label_batch = batch
        return (
            ImageBatch(
                image_batch.to(
                    self.device,
                    non_blocking=self.non_blocking,
                )
            ),
            [
                PascalBoxes(
                    x.to(
                        self.device,
                        non_blocking=self.non_blocking,
                    )
                )
                for x in boxes_batch
            ],
            [
                Labels(
                    x.to(
                        self.device,
                        non_blocking=self.non_blocking,
                    )
                )
                for x in label_batch
            ],
        )


class ToBoxes:
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        limit: int = 1000,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.limit = limit

    @torch.no_grad()
    def __call__(
        self, net_output: NetOutput
    ) -> t.Tuple[List[PascalBoxes], List[Confidences], List[Labels]]:
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

            box_list: List[Tensor] = []
            confidence_list: List[Tensor] = []
            label_list: List[Tensor] = []
            for c in unique_labels:
                cls_indices = labels == c
                if cls_indices.sum() == 0:
                    continue

                c_boxes = boxes[cls_indices]
                c_confidences = confidences[cls_indices]
                c_labels = labels[cls_indices]

                sort_indices = c_confidences.argsort(descending=True)[: self.limit]
                c_boxes = c_boxes[sort_indices]
                c_confidences = c_confidences[sort_indices]
                c_labels = c_labels[sort_indices]

                nms_indices = nms(
                    c_boxes,
                    c_confidences,
                    self.iou_threshold,
                )
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
            box_batch.append(PascalBoxes(boxes))
            confidence_batch.append(Confidences(confidences))
            label_batch.append(Labels(labels))
        return box_batch, confidence_batch, label_batch
