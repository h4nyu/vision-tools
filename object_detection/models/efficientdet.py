import numpy as np
import typing as t
import torch
import torch.nn.functional as F
import math
import torchvision

from object_detection.entities.image import ImageBatch
from object_detection.entities.box import Labels, YoloBoxes, PascalBoxes, yolo_to_pascal
from object_detection.entities import Batch, ImageId, Confidences
from object_detection.model_loader import ModelLoader
from object_detection.meters import BestWatcher, MeanMeter
from object_detection.utils import DetectionPlot
from typing import Any, List, Tuple
from torchvision.ops import nms
from torchvision.ops.boxes import box_iou
from torch.utils.data import DataLoader
from torch import nn, Tensor
from itertools import product as product
from logging import getLogger
from pathlib import Path
from typing_extensions import Literal

from .backbones import EfficientNetBackbone
from .bifpn import BiFPN, FP
from .losses import FocalLoss
from .anchors import Anchors

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


class Visualize:
    def __init__(
        self, out_dir: str, prefix: str, limit: int = 1, use_alpha: bool = True
    ) -> None:
        self.prefix = prefix
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.use_alpha = use_alpha

    def __call__(
        self,
        src: List[Tuple[ImageId, YoloBoxes, Confidences]],
        tgt: List[YoloBoxes],
        image_batch: ImageBatch,
    ) -> None:
        image_batch = ImageBatch(image_batch[: self.limit])
        src = src[: self.limit]
        tgt = tgt[: self.limit]
        _, _, h, w = image_batch.shape
        for i, ((_, sb, sc), tb, img) in enumerate(zip(src, tgt, image_batch)):
            plot = DetectionPlot(h=h, w=w, use_alpha=self.use_alpha)
            plot.with_image(img, alpha=0.5)
            plot.with_yolo_boxes(tb, color="blue")
            plot.with_yolo_boxes(sb, sc, color="red")
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")


def collate_fn(
    batch: Batch,
) -> Tuple[ImageBatch, List[YoloBoxes], List[Labels], List[ImageId]]:
    images: List[t.Any] = []
    id_batch: List[ImageId] = []
    box_batch: List[YoloBoxes] = []
    label_batch: List[Labels] = []

    for id, img, boxes, labels in batch:
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return ImageBatch(torch.stack(images)), box_batch, label_batch, id_batch


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
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):  # type: ignore
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
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
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.act2 = nn.ReLU()
        self.out = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.out_act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, 4)
        x = self.out_act(x)
        return x


NetOutput = Tuple[YoloBoxes, YoloBoxes, Tensor]


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: EfficientNetBackbone,
        channels: int = 32,
        threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.anchors = Anchors(size=2)
        self.backbone = backbone
        self.clip_boxes = ClipBoxes()
        self.neck = BiFPN(channels=channels)
        self.threshold = threshold
        self.regression = RegressionModel(
            in_channels=channels,
            hidden_channels=channels,
            num_anchors=self.anchors.num_anchors,
        )
        self.classification = ClassificationModel(
            channels, num_classes=num_classes, num_anchors=self.anchors.num_anchors
        )

    def forward(self, images: ImageBatch) -> NetOutput:
        features = self.backbone(images)
        anchors = torch.cat([self.anchors(i) for i in features], dim=0)
        box_diffs = torch.cat([self.regression(i) for i in features], dim=1)
        labels = torch.cat([self.classification(i) for i in features], dim=1)
        return YoloBoxes(anchors), YoloBoxes(box_diffs), Labels(labels)


class Criterion:
    def __init__(
        self,
        num_classes: int = 1,
        box_weight: float = 1.0,
        label_weight: float = 10.0,
    ) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.label_weight = label_weight
        self.label_loss = LabelLoss()
        self.box_loss = BoxLoss()

    def __call__(
        self,
        images: ImageBatch,
        net_output: NetOutput,
        gt_boxes_list: List[YoloBoxes],
        gt_classes_list: List[Labels],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        anchors, regressions, classifications = net_output
        device = classifications.device
        batch_size = classifications.shape[0]
        box_losses: List[Tensor] = []
        label_losses: List[Tensor] = []
        _, _, h, w = images.shape

        for pred_boxes, pred_labels, gt_boxes, gt_lables in zip(
            regressions, classifications, gt_boxes_list, gt_classes_list
        ):
            iou_matrix = box_iou(
                yolo_to_pascal(anchors, size=(w, h)),
                yolo_to_pascal(gt_boxes, size=(w, h)),
            )
            iou_max, match_indices = torch.max(iou_matrix, dim=1)
            label_losses.append(
                self.label_loss(
                    iou_max=iou_max,
                    match_indices=match_indices,
                    pred_classes=pred_labels,
                    gt_classes=gt_lables,
                )
            )
            box_losses.append(
                self.box_loss(
                    iou_max=iou_max,
                    match_indices=match_indices,
                    anchors=anchors,
                    pred_boxes=pred_boxes,
                    gt_boxes=gt_boxes,
                )
            )
        all_box_loss = torch.stack(box_losses).mean() * self.box_weight
        all_label_loss = torch.stack(label_losses).mean() * self.label_weight
        loss = all_box_loss + all_label_loss
        return loss, all_box_loss, all_label_loss


class BoxLoss:
    def __init__(self, iou_threshold: float = 0.6) -> None:
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
        if num_pos == 0:
            return torch.tensor(0.0).to(device)
        matched_gt_boxes = gt_boxes[match_indices][positive_indices]
        matched_anchors = anchors[positive_indices]
        matched_pred_boxes = pred_boxes[positive_indices]
        diffs = matched_gt_boxes - matched_anchors
        return F.l1_loss(matched_pred_boxes, diffs, reduction="none").sum() / num_pos


class LabelLoss:
    def __init__(self, iou_thresholds: Tuple[float, float] = (0.5, 0.55)) -> None:
        """
        focal_loss
        """
        self.gamma = 2.0
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

        targets = torch.ones(pred_classes.shape).to(device) * -1.0
        targets[positive_indices, matched_gt_classes[positive_indices].long()] = 1
        targets[negative_indices, :] = 0
        pred_classes = torch.clamp(pred_classes, min=1e-4, max=1 - 1e-4)
        pos_loss = (
            -((1 - pred_classes) ** self.gamma)
            * torch.log(pred_classes)
            * targets.eq(1.0)
        )
        pos_loss = pos_loss.sum() / positive_indices.sum().clamp(min=1)
        neg_loss = (
            -((pred_classes) ** self.gamma)
            * torch.log(1 - pred_classes)
            * targets.eq(0.0)
        )
        neg_loss = neg_loss.sum() / negative_indices.sum().clamp(min=1)
        loss = pos_loss + neg_loss
        return loss


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        visualize: Visualize,
        optimizer: t.Any,
        device: str = "cpu",
        criterion: Criterion = Criterion(),
    ) -> None:
        self.device = torch.device(device)
        self.model_loader = model_loader
        self.model = model_loader.model.to(self.device)
        self.preprocess = PreProcess(self.device)
        self.postprocess = PostProcess()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.best_watcher = BestWatcher()
        self.visualize = visualize
        self.meters = {
            key: MeanMeter()
            for key in [
                "train_loss",
                "train_box",
                "train_label",
                "test_loss",
                "test_box",
                "test_label",
            ]
        }

    def log(self) -> None:
        value = ("|").join([f"{k}:{v.get_value():.4f}" for k, v in self.meters.items()])
        logger.info(value)

    def reset_meters(self) -> None:
        for v in self.meters.values():
            v.reset()

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()
            self.eval_one_epoch()
            self.log()
            self.reset_meters()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for samples, gt_boxes_list, gt_labes_list, ids in loader:
            samples, gt_boxes_list, gt_labels_list = self.preprocess(
                (samples, gt_boxes_list, gt_labes_list)
            )
            outputs = self.model(samples)
            loss, box_loss, label_loss = self.criterion(
                samples, outputs, gt_boxes_list, gt_labes_list
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.meters["train_loss"].update(loss.item())
            self.meters["train_box"].update(box_loss.item())
            self.meters["train_label"].update(label_loss.item())

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for samples, gt_boxes_list, gt_labes_list, ids in loader:
            samples, gt_boxes_list, gt_labels_list = self.preprocess(
                (samples, gt_boxes_list, gt_labes_list)
            )
            outputs = self.model(samples)
            loss, box_loss, label_loss = self.criterion(
                samples, outputs, gt_boxes_list, gt_labes_list
            )
            self.meters["test_loss"].update(loss.item())
            self.meters["test_box"].update(box_loss.item())
            self.meters["test_label"].update(label_loss.item())
        preds = self.postprocess(outputs, ids, samples)
        self.visualize(preds, gt_boxes_list, samples)
        if self.best_watcher.step(self.meters["test_loss"].get_value()):
            self.model_loader.save({"loss": self.meters["test_loss"].get_value()})


class PreProcess:
    def __init__(self, device: t.Any) -> None:
        super().__init__()
        self.device = device

    def __call__(
        self, batch: t.Tuple[ImageBatch, List[YoloBoxes], List[Labels]]
    ) -> t.Tuple[ImageBatch, List[YoloBoxes], List[Labels]]:
        image_batch, boxes_batch, label_batch = batch
        return (
            ImageBatch(image_batch.to(self.device)),
            [YoloBoxes(x.to(self.device)) for x in boxes_batch],
            [Labels(x.to(self.device)) for x in label_batch],
        )


class PostProcess:
    def __init__(self, box_limit: int = 100) -> None:
        self.box_limit = box_limit

    def __call__(
        self, net_output: NetOutput, image_ids: List[ImageId], images: ImageBatch,
    ) -> List[Tuple[ImageId, YoloBoxes, Confidences]]:
        _, _, h, w = images.shape
        anchors, boxes_batch, labels_batch = net_output
        rows = []
        for image_id, diff_boxes, confidences in zip(
            image_ids, boxes_batch, labels_batch
        ):
            boxes = anchors + diff_boxes
            confidences, _ = confidences.max(dim=1)
            sort_idx = nms(
                yolo_to_pascal(YoloBoxes(boxes), (w, h)), confidences, iou_threshold=0.5
            )
            sort_idx = sort_idx[: self.box_limit]
            boxes = boxes[sort_idx]
            confidences = confidences[sort_idx]
            rows.append((image_id, YoloBoxes(boxes), Confidences(confidences)))
        return rows
