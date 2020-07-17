import numpy as np
import typing as t
import torch
import torch.nn.functional as F
import math
import torchvision

from object_detection.entities import (
    ImageId,
    Confidences,
    PyramidIdx,
    TrainSample,
    Labels,
    YoloBoxes,
    PascalBoxes,
    yolo_to_pascal,
    ImageBatch,
)
from object_detection.model_loader import ModelLoader
from object_detection.meters import MeanMeter
from object_detection.utils import DetectionPlot
from typing import Any, List, Tuple, NewType, Callable
from torchvision.ops.boxes import box_iou
from torch.utils.data import DataLoader
from torch import nn, Tensor
from itertools import product as product
from logging import getLogger
from pathlib import Path
from typing_extensions import Literal
from tqdm import tqdm
from .box_merge import BoxMerge

from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import FocalLoss
from .anchors import Anchors

logger = getLogger(__name__)


class Visualize:
    def __init__(
        self,
        out_dir: str,
        prefix: str,
        limit: int = 1,
        use_alpha: bool = True,
        show_probs: bool = True,
    ) -> None:
        self.prefix = prefix
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.use_alpha = use_alpha
        self.show_probs = show_probs

    def __call__(
        self,
        src: Tuple[List[YoloBoxes], List[Confidences]],
        tgt: List[YoloBoxes],
        image_batch: ImageBatch,
    ) -> None:
        image_batch = ImageBatch(image_batch[: self.limit])
        tgt = tgt[: self.limit]
        src_box_batch, src_confidence_batch = src
        _, _, h, w = image_batch.shape
        for i, (img, sb, sc, tb) in enumerate(
            zip(image_batch, src_box_batch, src_confidence_batch, tgt)
        ):
            plot = DetectionPlot(
                h=h, w=w, use_alpha=self.use_alpha, show_probs=self.show_probs
            )
            plot.with_image(img, alpha=0.5)
            plot.with_yolo_boxes(tb, color="blue")
            plot.with_yolo_boxes(sb, sc, color="red")
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")


def collate_fn(
    batch: List[TrainSample],
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
        in_channels: int,
        num_anchors: int = 9,
        num_classes: int = 80,
        prior: float = 0.01,
        feature_size: int = 256,
        depth: int = 1,
    ) -> None:
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_conv = SENextBottleneck2d(in_channels, feature_size)
        self.bottlenecks = nn.Sequential(
            *[SENextBottleneck2d(feature_size, feature_size) for _ in range(depth)]
        )
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.bottlenecks(x)
        x = self.output(x)
        x = self.output_act(x)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out = x.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out.contiguous().view(batch_size, -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_size: int = 2,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.in_conv = SENextBottleneck2d(in_channels, hidden_channels)
        self.bottlenecks = nn.Sequential(
            *[
                SENextBottleneck2d(hidden_channels, hidden_channels)
                for _ in range(depth)
            ]
        )
        self.out = nn.Conv2d(
            hidden_channels, num_anchors * self.out_size, kernel_size=3, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.bottlenecks(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, self.out_size)
        return x


PosDiff = NewType("PosDiff", Tensor)
PosDiffs = NewType("PosDiffs", Tensor)
SizeDiff = NewType("SizeDiff", Tensor)
SizeDiffs = NewType("SizeDiffs", Tensor)

NetOutput = Tuple[YoloBoxes, PosDiffs, SizeDiffs, Tensor]


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        channels: int = 32,
        threshold: float = 0.01,
        out_ids: List[PyramidIdx] = [4, 5, 6],
        anchors: Anchors = Anchors(size=2),
    ) -> None:
        super().__init__()
        self.out_ids = np.array(out_ids) - 3
        self.anchors = anchors
        self.backbone = backbone
        self.neck = BiFPN(channels=channels)
        self.pos_reg = RegressionModel(
            in_channels=channels,
            hidden_channels=channels,
            num_anchors=self.anchors.num_anchors,
        )
        self.size_reg = RegressionModel(
            in_channels=channels,
            hidden_channels=channels,
            num_anchors=self.anchors.num_anchors,
        )
        self.classification = ClassificationModel(
            channels, num_classes=num_classes, num_anchors=self.anchors.num_anchors
        )

    def forward(self, images: ImageBatch) -> NetOutput:
        features = self.backbone(images)
        features = self.neck(features)
        anchors = torch.cat([self.anchors(features[i]) for i in self.out_ids], dim=0)
        pos_diffs = torch.cat([self.pos_reg(features[i]) for i in self.out_ids], dim=1)
        size_diffs = torch.cat(
            [self.size_reg(features[i]) for i in self.out_ids], dim=1
        )
        labels = torch.cat(
            [self.classification(features[i]) for i in self.out_ids], dim=1
        )
        return (
            YoloBoxes(anchors),
            PosDiffs(pos_diffs),
            SizeDiffs(size_diffs),
            Labels(labels),
        )


class Criterion:
    def __init__(
        self,
        num_classes: int = 1,
        pos_weight: float = 4.0,
        size_weight: float = 1.0,
        label_weight: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.pos_weight = pos_weight
        self.size_weight = size_weight
        self.label_weight = label_weight
        self.label_loss = LabelLoss()
        self.pos_loss = PosLoss()
        self.size_loss = SizeLoss()

    def __call__(
        self,
        images: ImageBatch,
        net_output: NetOutput,
        gt_boxes_list: List[YoloBoxes],
        gt_classes_list: List[Labels],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        anchors, pos_diffs, size_diffs, classifications = net_output
        device = classifications.device
        batch_size = classifications.shape[0]
        pos_losses: List[Tensor] = []
        size_losses: List[Tensor] = []
        label_losses: List[Tensor] = []
        _, _, h, w = images.shape

        for pos_diff, size_diff, pred_labels, gt_boxes, gt_lables in zip(
            pos_diffs, size_diffs, classifications, gt_boxes_list, gt_classes_list
        ):
            if len(gt_boxes) == 0:
                continue
            iou_matrix = box_iou(
                yolo_to_pascal(anchors, wh=(w, h)), yolo_to_pascal(gt_boxes, wh=(w, h)),
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
            pos_losses.append(
                self.pos_loss(
                    iou_max=iou_max,
                    match_indices=match_indices,
                    anchors=anchors,
                    pos_diff=PosDiff(pos_diff),
                    gt_boxes=gt_boxes,
                )
            )
            size_losses.append(
                self.size_loss(
                    iou_max=iou_max,
                    match_indices=match_indices,
                    anchors=anchors,
                    size_diff=SizeDiff(size_diff),
                    gt_boxes=gt_boxes,
                )
            )
        all_pos_loss = torch.stack(pos_losses).mean() * self.pos_weight
        all_size_loss = torch.stack(size_losses).mean() * self.size_weight
        all_label_loss = torch.stack(label_losses).mean() * self.label_weight
        loss = all_pos_loss + all_size_loss + all_label_loss
        return loss, all_pos_loss, all_size_loss, all_label_loss


class SizeLoss:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        iou_max: Tensor,
        match_indices: Tensor,
        anchors: Tensor,
        size_diff: SizeDiff,
        gt_boxes: YoloBoxes,
    ) -> Tensor:
        device = size_diff.device
        high = self.iou_threshold
        positive_indices = iou_max > self.iou_threshold
        num_pos = positive_indices.sum()
        if num_pos == 0:
            return torch.tensor(0.0).to(device)
        matched_gt_boxes = gt_boxes[match_indices][positive_indices][:, 2:]
        matched_anchors = anchors[positive_indices][:, 2:]
        pred_diff = size_diff[positive_indices]
        gt_diff = torch.log(matched_gt_boxes / matched_anchors)
        return F.l1_loss(pred_diff, gt_diff, reduction="mean")


class PosLoss:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        iou_max: Tensor,
        match_indices: Tensor,
        anchors: Tensor,
        pos_diff: PosDiff,
        gt_boxes: YoloBoxes,
    ) -> Tensor:
        device = pos_diff.device
        high = self.iou_threshold
        positive_indices = iou_max > self.iou_threshold
        num_pos = positive_indices.sum()
        if num_pos == 0:
            return torch.tensor(0.0).to(device)
        matched_gt_boxes = gt_boxes[match_indices][positive_indices][:, :2]
        matched_anchor_pos = anchors[positive_indices][:, :2]
        matched_anchor_size = anchors[positive_indices][:, 2:]
        pred_diff = pos_diff[positive_indices]
        gt_diff = (matched_gt_boxes - matched_anchor_pos) / matched_anchor_size
        return F.l1_loss(pred_diff, gt_diff, reduction="mean")


class LabelLoss:
    def __init__(self, iou_thresholds: Tuple[float, float] = (0.4, 0.5)) -> None:
        """
        focal_loss
        """
        self.gamma = 2.0
        self.alpha = 0.25
        self.beta = 4.0
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
        negative_indices = iou_max <= low
        matched_gt_classes = gt_classes[match_indices]
        targets = torch.ones(pred_classes.shape).to(device) * -1.0
        targets[positive_indices, matched_gt_classes[positive_indices].long()] = 1

        targets[negative_indices, :] = 0
        pred_classes = torch.clamp(pred_classes, min=1e-4, max=1 - 1e-4)
        pos_loss = (
            -self.alpha
            * ((1 - pred_classes) ** self.gamma)
            * torch.log(pred_classes)
            * targets.eq(1.0)
        )
        pos_loss = pos_loss.sum()
        neg_loss = (
            -(1 - self.alpha)
            * ((pred_classes) ** self.gamma)
            * torch.log(1 - pred_classes)
            * targets.eq(0.0)
        )
        neg_loss = (neg_loss).sum()
        loss = (pos_loss + neg_loss) / positive_indices.sum().clamp(min=1.0)
        return loss


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


class ToBoxes:
    def __init__(self, confidence_threshold: float = 0.5, limit: int = 100,) -> None:
        self.confidence_threshold = confidence_threshold
        self.limit = limit

    def __call__(
        self, net_output: NetOutput
    ) -> t.Tuple[List[YoloBoxes], List[Confidences]]:
        anchors, pos_diffs, size_diffs, labels_batch = net_output
        box_batch = []
        confidence_batch = []
        for pos_diff, size_diff, confidences in zip(
            pos_diffs, size_diffs, labels_batch
        ):
            box_pos = anchors[:, 2:] * pos_diff + anchors[:, :2]
            box_size = anchors[:, 2:] * size_diff.exp()
            boxes = torch.cat([box_pos, box_size], dim=1)
            confidences, c_index = confidences.max(dim=1)

            filter_idx = confidences > self.confidence_threshold
            confidences = confidences[filter_idx]
            boxes = boxes[filter_idx]
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            boxes = boxes[sort_idx]
            confidences = confidences[sort_idx]
            box_batch.append(YoloBoxes(boxes))
            confidence_batch.append(Confidences(confidences))
        return box_batch, confidence_batch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        visualize: Visualize,
        optimizer: t.Any,
        get_score: Callable[[YoloBoxes, YoloBoxes], float],
        to_boxes: ToBoxes,
        box_merge: BoxMerge,
        device: str = "cpu",
        criterion: Criterion = Criterion(),
    ) -> None:
        self.device = torch.device(device)
        self.model_loader = model_loader
        self.model = model.to(self.device)
        self.preprocess = PreProcess(self.device)
        self.to_boxes = to_boxes
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.visualize = visualize
        self.get_score = get_score
        self.box_merge = box_merge
        self.meters = {
            key: MeanMeter()
            for key in [
                "train_loss",
                "train_pos",
                "train_size",
                "train_label",
                "test_loss",
                "test_pos",
                "test_size",
                "test_label",
                "score",
            ]
        }

    def log(self) -> None:
        value = ("|").join([f"{k}:{v.get_value():.4f}" for k, v in self.meters.items()])
        logger.info(value)

    def reset_meters(self) -> None:
        for v in self.meters.values():
            v.reset()

    def __call__(self, epochs: int) -> None:
        self.model = self.model_loader.load_if_needed(self.model)
        for epoch in range(epochs):
            self.train_one_epoch()
            self.eval_one_epoch()
            self.log()
            self.reset_meters()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for samples, gt_boxes_list, gt_labes_list, ids in tqdm(loader):
            samples, gt_boxes_list, gt_labels_list = self.preprocess(
                (samples, gt_boxes_list, gt_labes_list)
            )
            outputs = self.model(samples)
            loss, pos_loss, size_loss, label_loss = self.criterion(
                samples, outputs, gt_boxes_list, gt_labes_list
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.meters["train_loss"].update(loss.item())
            self.meters["train_pos"].update(pos_loss.item())
            self.meters["train_size"].update(size_loss.item())
            self.meters["train_label"].update(label_loss.item())
        preds = self.to_boxes(outputs)
        self.visualize(preds, gt_boxes_list, samples)

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.train()
        loader = self.test_loader
        for samples, box_batch, gt_labes_list, ids in tqdm(loader):
            samples, box_batch, gt_labels_list = self.preprocess(
                (samples, box_batch, gt_labes_list)
            )
            outputs = self.model(samples)
            loss, pos_loss, size_loss, label_loss = self.criterion(
                samples, outputs, box_batch, gt_labes_list
            )
            self.meters["test_loss"].update(loss.item())
            self.meters["test_pos"].update(pos_loss.item())
            self.meters["test_size"].update(size_loss.item())
            self.meters["test_label"].update(label_loss.item())
            preds = self.box_merge(self.to_boxes(outputs))
            for (pred, gt) in zip(preds[0], box_batch):
                self.meters["score"].update(self.get_score(pred, gt))

        self.visualize(preds, box_batch, samples)
        self.model_loader.save_if_needed(
            self.model, self.meters[self.model_loader.key].get_value()
        )
