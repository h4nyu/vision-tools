import numpy as np
import typing as t
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from object_detection.entities import (
    ImageId,
    Confidences,
    PyramidIdx,
    TrainSample,
    PredictionSample,
    Labels,
    YoloBoxes,
    PascalBoxes,
    yolo_to_pascal,
    ImageBatch,
    yolo_clamp,
)
from object_detection.model_loader import ModelLoader
from object_detection.meters import MeanMeter
from object_detection.utils import DetectionPlot
from .losses import DIoU, HuberLoss
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

from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import FocalLoss
from .anchors import Anchors
from .tta import VFlipTTA, HFlipTTA

logger = getLogger(__name__)


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


class ClassificationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 9,
        num_classes: int = 80,
        prior: float = 0.01,
        hidden_channels: int = 256,
        depth: int = 1,
    ) -> None:
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_conv = SENextBottleneck2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
        )
        self.bottlenecks = nn.Sequential(
            *[
                SENextBottleneck2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Conv2d(
            hidden_channels, num_anchors * num_classes, kernel_size=1, padding=0
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
        out_size: int = 4,
        num_anchors: int = 9,
        hidden_channels: int = 256,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.in_conv = SENextBottleneck2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
        )
        self.bottlenecks = nn.Sequential(
            *[
                SENextBottleneck2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                )
                for _ in range(depth)
            ]
        )
        self.out = nn.Conv2d(
            hidden_channels, num_anchors * self.out_size, kernel_size=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.bottlenecks(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, self.out_size)
        return x


BoxDiff = NewType("BoxDiff", Tensor)
BoxDiffs = NewType("BoxDiffs", Tensor)

NetOutput = Tuple[YoloBoxes, BoxDiffs, Tensor]


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        channels: int = 64,
        out_ids: List[PyramidIdx] = [3, 4, 5, 6],
        anchors: Anchors = Anchors(),
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.out_ids = np.array(out_ids) - 3
        self.anchors = anchors
        self.backbone = backbone
        self.neck = nn.Sequential(*[BiFPN(channels=channels) for _ in range(depth)])
        self.box_reg = RegressionModel(
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
        box_diffs = torch.cat([self.box_reg(features[i]) for i in self.out_ids], dim=1)
        labels = torch.cat(
            [self.classification(features[i]) for i in self.out_ids], dim=1
        )
        return (
            YoloBoxes(anchors),
            BoxDiffs(box_diffs),
            Labels(labels),
        )


class BoxLoss:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        #  self.loss = HuberLoss(delta=0.1)
        self.loss = partial(F.l1_loss, reduction="mean")

    def __call__(
        self,
        match_score: Tensor,
        match_indices: Tensor,
        anchors: Tensor,
        box_diff: BoxDiff,
        gt_boxes: YoloBoxes,
    ) -> Tensor:
        device = box_diff.device
        high = self.iou_threshold
        positive_indices = match_score < self.iou_threshold
        num_pos = positive_indices.sum()
        if num_pos == 0:
            return torch.tensor(0.0).float().to(device)
        matched_gt_boxes = gt_boxes[match_indices][positive_indices]
        matched_anchors = anchors[positive_indices]
        pred_diff = box_diff[positive_indices]
        gt_diff = matched_gt_boxes - matched_anchors
        return self.loss(pred_diff, gt_diff)


class LabelLoss:
    def __init__(self, iou_thresholds: Tuple[float, float] = (0.5, 0.6)) -> None:
        """
        focal_loss
        """
        self.gamma = 2.0
        self.alpha = 0.25
        self.beta = 4.0
        self.iou_thresholds = iou_thresholds

    def __call__(
        self,
        match_score: Tensor,
        match_indices: Tensor,
        pred_classes: Tensor,
        gt_classes: Tensor,
    ) -> Tensor:
        device = pred_classes.device
        low, high = self.iou_thresholds
        positive_indices = match_score <= low
        negative_indices = match_score > high
        matched_gt_classes = gt_classes[match_indices]
        targets = torch.ones(pred_classes.shape).to(device) * -1.0
        targets[positive_indices, matched_gt_classes[positive_indices].long()] = 1
        targets[negative_indices, :] = 0
        pred_classes = torch.clamp(pred_classes, min=1e-4, max=1 - 1e-4)
        pos_count = positive_indices.sum()
        if pos_count == 0:
            logger.debug("no box matched")
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
        loss = (pos_loss + neg_loss) / pos_count.clamp(min=1.0)
        return loss


class Criterion:
    def __init__(
        self,
        num_classes: int = 1,
        box_weight: float = 10.0,
        label_weight: float = 1.0,
        box_loss: BoxLoss = BoxLoss(),
        label_loss: LabelLoss = LabelLoss(),
    ) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.label_weight = label_weight
        self.label_loss = label_loss
        self.box_loss = box_loss
        self.diou = DIoU()

    def __call__(
        self,
        images: ImageBatch,
        net_output: NetOutput,
        gt_boxes_list: List[YoloBoxes],
        gt_classes_list: List[Labels],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        anchors, box_diffs, classifications = net_output
        device = classifications.device
        batch_size = classifications.shape[0]
        box_losses: List[Tensor] = []
        label_losses: List[Tensor] = []
        _, _, h, w = images.shape

        for box_diff, pred_labels, gt_boxes, gt_lables in zip(
            box_diffs, classifications, gt_boxes_list, gt_classes_list
        ):
            if len(gt_boxes) == 0:
                continue
            iou_matrix = self.diou(
                yolo_to_pascal(anchors, wh=(w, h)), yolo_to_pascal(gt_boxes, wh=(w, h)),
            )
            match_score, match_indices = torch.min(iou_matrix, dim=1)
            label_losses.append(
                self.label_loss(
                    match_score=match_score,
                    match_indices=match_indices,
                    pred_classes=pred_labels,
                    gt_classes=gt_lables,
                )
            )
            box_losses.append(
                self.box_loss(
                    match_score=match_score,
                    match_indices=match_indices,
                    anchors=anchors,
                    box_diff=BoxDiff(box_diff),
                    gt_boxes=gt_boxes,
                )
            )
        all_box_loss = torch.stack(box_losses).mean() * self.box_weight
        all_label_loss = torch.stack(label_losses).mean() * self.label_weight
        loss = all_box_loss + all_label_loss
        return loss, all_box_loss, all_label_loss


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
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        limit: int = 1000,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.limit = limit

    def __call__(
        self, net_output: NetOutput
    ) -> t.Tuple[List[YoloBoxes], List[Confidences]]:
        anchors, box_diffs, labels_batch = net_output
        box_batch = []
        confidence_batch = []
        for box_diff, confidences in zip(box_diffs, labels_batch):
            boxes = anchors + box_diff
            confidences, c_index = confidences.max(dim=1)
            filter_idx = confidences > self.confidence_threshold
            confidences = confidences[filter_idx][: self.limit]
            boxes = boxes[filter_idx][: self.limit]
            sort_idx = nms(
                yolo_to_pascal(boxes, (1, 1)), confidences, self.iou_threshold
            )
            confidences.argsort(descending=True)
            boxes = YoloBoxes(boxes[sort_idx])
            boxes = yolo_clamp(boxes)
            confidences = confidences[sort_idx]
            box_batch.append(boxes)
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
        self.meters = {
            key: MeanMeter()
            for key in [
                "train_loss",
                "train_box",
                "train_label",
                "test_loss",
                "test_box",
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
        loader = self.test_loader
        for samples, box_batch, gt_labes_list, ids in tqdm(loader):
            samples, box_batch, gt_labels_list = self.preprocess(
                (samples, box_batch, gt_labes_list)
            )
            outputs = self.model(samples)
            loss, box_loss, label_loss = self.criterion(
                samples, outputs, box_batch, gt_labes_list
            )
            self.meters["test_loss"].update(loss.item())
            self.meters["test_box"].update(box_loss.item())
            self.meters["test_label"].update(label_loss.item())

            preds = self.to_boxes(outputs)
            for (pred, gt) in zip(preds[0], box_batch):
                self.meters["score"].update(self.get_score(pred, gt))

        self.visualize(preds, box_batch, samples)
        self.model_loader.save_if_needed(
            self.model, self.meters[self.model_loader.key].get_value()
        )


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        model_loader: ModelLoader,
        to_boxes: ToBoxes,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model_loader = model_loader
        self.model = model.to(self.device)
        self.preprocess = PreProcess(self.device)
        self.to_boxes = to_boxes
        self.loader = loader

    @torch.no_grad()
    def __call__(self) -> Tuple[List[YoloBoxes], List[Confidences], List[ImageId]]:
        self.model = self.model_loader.load_if_needed(self.model)
        self.model.eval()
        boxes_list = []
        confs_list = []
        id_list = []
        for images, ids in tqdm(self.loader):
            images = images.to(self.device)
            preds = self.model(images)
            boxes_list += preds[0]
            confs_list += preds[1]
            id_list += ids
        return boxes_list, confs_list, id_list
