import numpy as np
import typing as t
import torch
import torch.nn.functional as F
import math
import torchvision

from object_detection.entities.image import ImageBatch
from object_detection.entities.box import Labels, YoloBoxes, PascalBoxes, box_iou
from typing import Any, List, Tuple
from torchvision.ops import nms
from torch import nn, Tensor
from itertools import product as product
from .backbones import EfficientNetBackbone
from logging import getLogger
from .bifpn import BiFPN, FP
from .losses import FocalLoss
from .anchors import Anchors
from typing_extensions import Literal

logger = getLogger(__name__)

ModelName = Literal[
    "efficientdet-d0",
    "efficientdet-d1",
    "efficientdet-d2",
    "efficientdet-d3",
    "efficientdet-d4",
    "efficientdet-d5",
    "efficientdet-d6",
    "efficientdet-d7",
]


class BBoxTransform(nn.Module):
    def __init__(
        self, mean: t.Optional[t.Any] = None, std: t.Optional[t.Any] = None
    ) -> None:
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            )
        else:
            self.std = std

    def forward(self, boxes, deltas):  # type: ignore

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
        #  print(dx, dy,dw,dh)

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(
        self, width: t.Optional[int] = None, height: t.Optional[int] = None
    ) -> None:
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):  # type: ignore
        """
        boxes: [B, C, 4] [xmin, ymin, xmax, ymax]
        img: [B, C, W, H]
        """

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_anchors: int = 9,
        num_classes: int = 80,
        prior: float = 0.01,
        feature_size: int = 256,
    ) -> None:
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):  # type: ignore
        out = self.conv1(x)
        out = self.act1(out)
        out = self.output(out)
        out = self.output_act(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(
        self, in_channels: int, num_anchors: int = 9, hidden_channels: int = 256
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.out = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        return x.contiguous().view(x.shape[0], -1, 4)


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: EfficientNetBackbone,
        channels: int = 32,
        threshold: float = 0.01,
        iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.anchors = Anchors()
        self.backbone = backbone
        self.clip_boxes = ClipBoxes()
        self.neck = BiFPN(channels=channels)
        self.threshold = threshold
        self.regression = RegressionModel(
            in_channels=channels, hidden_channels=channels
        )
        self.classification = ClassificationModel(channels, num_classes=2)
        self.bbox_transform = BBoxTransform()
        self.iou_threshold = iou_threshold

    def forward(
        self, images: ImageBatch, annotations: t.Optional[Tensor] = None
    ) -> Any:
        features = self.backbone(images)
        for i in features:
            print(i.shape)

        print(self.regression(features[0]).shape)
        #  regressions = torch.cat(
        #      [self.regression(feature) for feature in features], dim=1
        #  )
        #  classifications = torch.cat(
        #      [self.classification(feature) for feature in features], dim=1
        #  )
        #  print(regressions.shape)
        #  anchors = self.anchors(images)
        #  print(anchors.shape)
        #  return classifications, regressions

        #  if annotations is not None:
        #      return self.criterion(classifications, regressions, anchors, annotations)

        #  if annotations is None:
        #      transformed_anchors = self.bbox_transform(anchors, regressions)
        #      transformed_anchors = self.clip_boxes(transformed_anchors, inputs)
        #      scores = torch.max(classifications, dim=2, keepdim=True)[0]
        #
        #      scores_over_thresh = (scores > self.threshold)[0, :, 0]
        #      if scores_over_thresh.sum() == 0:
        #          logger.info("No boxes to NMS")
        #          return torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)
        #      classification = classifications[:, scores_over_thresh, :]
        #      anchors_nms_idx = nms(
        #          transformed_anchors[0, :, :],
        #          scores[0, :, 0],
        #          iou_threshold=self.iou_threshold,
        #      )
        #      nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)
        #      return nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]


class Criterion:
    def __init__(self, iou_threshold: float = 0.5, num_classes: int = 1,) -> None:
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.label_loss = LabelLoss()
        self.box_loss = BoxLoss()

    def __call__(
        self,
        classifications: Tensor,
        regressions: YoloBoxes,
        anchors: YoloBoxes,
        gt_boxes_list: List[YoloBoxes],
        gt_classes_list: List[Labels],
    ) -> Tensor:
        device = classifications.device
        batch_size = classifications.shape[0]
        box_losses: List[Tensor] = []
        label_losses: List[Tensor] = []

        # foreach batch item
        for pred_boxes, pred_labels, gt_boxes, gt_lables in zip(
            regressions, classifications, gt_boxes_list, gt_classes_list
        ):
            if len(gt_boxes) == 0:
                box_losses.append(torch.tensor(0).to(device))
                continue
            iou_matrix = box_iou(anchors, gt_boxes)
            iou_max, match_indices = torch.max(iou_matrix, dim=1)
            label_losses.append(
                self.label_loss(
                    iou_max=iou_max,
                    match_indices=match_indices,
                    pred_classes=pred_labels,
                    gt_classes=gt_lables,
                )
            )

        return torch.tensor(0)

        #  for pred_class, pred_boxes, gt_class, gt_boxes in zip(classifications, regressions)
        #
        #      bbox_annotation = annotations[j, :, :]
        #      bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
        #
        #      if bbox_annotation.shape[0] == 0:
        #          regression_losses.append(torch.tensor(0).float().cuda())
        #          classification_losses.append(torch.tensor(0).float().cuda())
        #
        #          continue
        #
        #      classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
        #
        #      # num_anchors x num_annotations
        #      IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
        #
        #      IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
        #
        #      # import pdb
        #      # pdb.set_trace()
        #
        #      # compute the loss for classification
        #      targets = torch.ones(classification.shape) * -1
        #      targets = targets.cuda()
        #
        #      targets[torch.lt(IoU_max, 0.4), :] = 0
        #
        #      positive_indices = torch.ge(IoU_max, 0.5)
        #
        #      num_positive_anchors = positive_indices.sum()
        #
        #      assigned_annotations = bbox_annotation[IoU_argmax, :]
        #
        #      targets[positive_indices, :] = 0
        #      targets[
        #          positive_indices, assigned_annotations[positive_indices, 4].long()
        #      ] = 1
        #
        #      alpha_factor = torch.ones(targets.shape).cuda() * alpha
        #
        #      alpha_factor = torch.where(
        #          torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor
        #      )
        #      focal_weight = torch.where(
        #          torch.eq(targets, 1.0), 1.0 - classification, classification
        #      )
        #      focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
        #
        #      bce = -(
        #          targets * torch.log(classification)
        #          + (1.0 - targets) * torch.log(1.0 - classification)
        #      )
        #
        #      # cls_loss = focal_weight * torch.pow(bce, gamma)
        #      cls_loss = focal_weight * bce
        #
        #      cls_loss = torch.where(
        #          torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()
        #      )
        #
        #      classification_losses.append(
        #          cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
        #      )
        #
        #      # compute the loss for regression
        #
        #      if positive_indices.sum() > 0:
        #          assigned_annotations = assigned_annotations[positive_indices, :]
        #
        #          anchor_widths_pi = anchor_widths[positive_indices]
        #          anchor_heights_pi = anchor_heights[positive_indices]
        #          anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
        #          anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
        #
        #          gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        #          gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        #          gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
        #          gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
        #
        #          # clip widths to 1
        #          gt_widths = torch.clamp(gt_widths, min=1)
        #          gt_heights = torch.clamp(gt_heights, min=1)
        #
        #          targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
        #          targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
        #          targets_dw = torch.log(gt_widths / anchor_widths_pi)
        #          targets_dh = torch.log(gt_heights / anchor_heights_pi)
        #
        #          targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
        #          targets = targets.t()
        #
        #          targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
        #
        #          negative_indices = 1 + (~positive_indices)
        #
        #          regression_diff = torch.abs(targets - regression[positive_indices, :])
        #
        #          regression_loss = torch.where(
        #              torch.le(regression_diff, 1.0 / 9.0),
        #              0.5 * 9.0 * torch.pow(regression_diff, 2),
        #              regression_diff - 0.5 / 9.0,
        #          )
        #          regression_losses.append(regression_loss.mean())
        #      else:
        #          regression_losses.append(torch.tensor(0).float().cuda())
        #
        #  return (
        #      torch.stack(classification_losses).mean(dim=0, keepdim=True),
        #      torch.stack(regression_losses).mean(dim=0, keepdim=True),
        #  )


class BoxLoss:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        wrap focal_loss
        """
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        iou_max: Tensor,
        match_indices: Tensor,
        anchors: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
    ) -> Tensor:
        device = pred_boxes.device
        high = self.iou_threshold
        positive_indices = iou_max > high
        num_pos = positive_indices.sum()
        matched_gt_boxes = gt_boxes[match_indices][positive_indices]
        matched_anchors = anchors[positive_indices]
        matched_pred_boxes = pred_boxes[positive_indices]
        diffs = matched_gt_boxes - matched_anchors
        return F.l1_loss(matched_pred_boxes, diffs, reduction="none").sum() / num_pos


class LabelLoss:
    def __init__(self, iou_thresholds: Tuple[float, float] = (0.4, 0.5)) -> None:
        """
        wrap focal_loss
        """

        self.focal_loss = FocalLoss()
        self.iou_thresholds = iou_thresholds

    def __call__(
        self,
        iou_max: Tensor,
        match_indices: Tensor,
        pred_classes: Tensor,
        gt_classes: Tensor,
    ) -> Tensor:
        device = pred_classes.device
        low, high = self.iou_thresholds
        positive_indices = iou_max > high
        ignore_indecices = (iou_max >= low) & (iou_max <= high)
        negative_indices = iou_max < low
        matched_gt_classes = gt_classes[match_indices]

        targets = torch.zeros(pred_classes.shape).to(device)
        targets[positive_indices, matched_gt_classes[positive_indices]] = 1
        pred_classes[ignore_indecices] = 0
        return self.focal_loss(pred_classes, targets)
