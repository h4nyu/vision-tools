import torch
import numpy as np
import math
import typing as t
import torch as tr
import torch.nn.functional as F
from functools import partial
from .activations import FReLU
from typing import (
    List,
    Tuple,
    NewType,
    Union,
    Callable,
    Any,
)
from torch import nn, Tensor
from typing_extensions import Literal
from logging import getLogger
from tqdm import tqdm
from object_detection.entities import (
    BoxMap,
    BoxMaps,
    YoloBoxes,
    Confidences,
    PascalBoxes,
    yolo_to_pascal,
    pascal_to_yolo,
    yolo_to_coco,
    Labels,
    boxmap_to_boxes,
)
from object_detection.utils import DetectionPlot
from object_detection.entities.image import ImageId
from .mkmaps import Heatmaps, MkMapsFn, MkBoxMapsFn
from .modules import ConvBR2d, Mish
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import HuberLoss, DIoULoss
from .anchors import EmptyAnchors
from .matcher import NearnestMatcher, CenterMatcher
from object_detection.meters import MeanMeter
from object_detection.entities import (
    ImageBatch,
    PredBoxes,
    Image,
    TrainSample,
)
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import nms
from torch.utils.data import DataLoader
from object_detection.model_loader import ModelLoader

from pathlib import Path

logger = getLogger(__name__)


def collate_fn(
    batch: List[TrainSample],
) -> Tuple[List[ImageId], ImageBatch, List[YoloBoxes], List[Labels]]:
    images: List[t.Any] = []
    id_batch: List[ImageId] = []
    box_batch: List[YoloBoxes] = []
    label_batch: List[Labels] = []

    for id, img, boxes, labels in batch:
        images.append(img)
        _, h, w = img.shape
        box_batch.append(pascal_to_yolo(boxes, (w, h)))
        id_batch.append(id)
        label_batch.append(labels)
    return (
        id_batch,
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
    )


class Reg(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(*[FReLU(in_channels) for _ in range(depth)])

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.out(x)
        return x


NetOutput = Tuple[Heatmaps, BoxMaps, BoxMap]  # label, pos, size, count


class CenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        backbone: nn.Module,
        depth: int = 2,
        out_idx: int = 4,
    ) -> None:
        super().__init__()
        self.out_idx = out_idx - 3
        self.channels = channels
        self.backbone = backbone
        self.fpn = nn.Sequential(*[BiFPN(channels=channels) for _ in range(depth)])
        self.hm_reg = nn.Sequential(
            Reg(
                in_channels=channels,
                out_channels=num_classes,
                depth=depth,
            ),
            nn.Sigmoid(),
        )
        self.box_reg = nn.Sequential(
            Reg(
                in_channels=channels,
                out_channels=4,
                depth=depth,
            ),
            nn.Sigmoid(),
        )
        self.anchors = EmptyAnchors()

    def forward(self, x: ImageBatch) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmaps = Heatmaps(self.hm_reg(fp[self.out_idx]))
        anchors = self.anchors(heatmaps)
        boxmaps = self.box_reg(fp[self.out_idx])
        return heatmaps, BoxMaps(boxmaps), anchors


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
        images: ImageBatch,
        netout: NetOutput,
        gt_box_batch: t.List[YoloBoxes],
        gt_label_batch: t.List[Labels],
    ) -> Tuple[Tensor, Tensor, Tensor, Heatmaps]:
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
        preds: BoxMaps,
        gt_box_batch: List[YoloBoxes],
        anchormap: BoxMap,
    ) -> Tensor:
        device = preds.device
        _, _, h, w = preds.shape
        box_losses: List[Tensor] = []
        anchors = boxmap_to_boxes(anchormap)
        for diff_map, gt_boxes in zip(preds, gt_box_batch):
            if len(gt_boxes) == 0:
                continue

            pred_boxes = boxmap_to_boxes(BoxMap(diff_map))
            match_indices, positive_indices = self.matcher(anchors, gt_boxes, (w, h))
            num_pos = positive_indices.sum()
            if num_pos == 0:
                continue
            matched_gt_boxes = YoloBoxes(gt_boxes[match_indices][positive_indices])
            matched_pred_boxes = YoloBoxes(pred_boxes[positive_indices])
            if self.use_diff:
                matched_pred_boxes = YoloBoxes(
                    anchors[positive_indices] + matched_pred_boxes
                )
            box_losses.append(
                self.loss(
                    yolo_to_pascal(matched_pred_boxes, (1, 1)),
                    yolo_to_pascal(matched_gt_boxes, (1, 1)),
                )
            )
        if len(box_losses) == 0:
            return torch.tensor(0.0).to(device)
        return torch.stack(box_losses).mean()


class ToBoxes:
    def __init__(
        self,
        threshold: float = 0.1,
        kernel_size: int = 3,
        limit: int = 1000,
        use_diff: bool = True,
    ) -> None:
        self.limit = limit
        self.threshold = threshold
        self.kernel_size = kernel_size
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
    ) -> t.Tuple[t.List[YoloBoxes], t.List[Confidences], t.List[Labels]]:
        heatmaps, boxmaps, anchormap = inputs
        device = heatmaps.device
        kpmaps = heatmaps * (
            (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        )
        kpmaps, labelmaps = torch.max(kpmaps, dim=1)

        box_batch: t.List[YoloBoxes] = []
        confidence_batch: t.List[Confidences] = []
        label_batch: t.List[Labels] = []
        for km, lm, bm in zip(kpmaps, labelmaps, boxmaps):
            kp = torch.nonzero(km, as_tuple=False)
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
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            box_batch.append(YoloBoxes(boxes[sort_idx]))
            confidence_batch.append(Confidences(confidences[sort_idx]))
            label_batch.append(Labels(labels[sort_idx]))
        return box_batch, confidence_batch, label_batch



class Visualize:
    def __init__(
        self,
        out_dir: str,
        prefix: str,
        limit: int = 1,
        use_alpha: bool = True,
        show_confidences: bool = True,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        self.prefix = prefix
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.use_alpha = use_alpha
        self.show_confidences = show_confidences
        self.figsize = figsize

    @torch.no_grad()
    def __call__(
        self,
        net_out: NetOutput,
        box_batch: List[YoloBoxes],
        confidence_batch: List[Confidences],
        label_batch: List[Labels],
        gt_box_batch: List[YoloBoxes],
        gt_label_batch: List[Labels],
        image_batch: ImageBatch,
        gt_hms: Heatmaps,
    ) -> None:
        heatmap, _, _ = net_out
        box_batch = box_batch[: self.limit]
        confidence_batch = confidence_batch[: self.limit]
        label_batch = label_batch[: self.limit]
        gt_box_batch = gt_box_batch[: self.limit]
        gt_label_batch = gt_label_batch[: self.limit]
        _, _, h, w = image_batch.shape
        for i, (
            boxes,
            confidences,
            labels,
            gt_boxes,
            gt_labels,
            hm,
            img,
            gt_hm,
        ) in enumerate(
            zip(
                box_batch,
                confidence_batch,
                label_batch,
                gt_box_batch,
                gt_label_batch,
                heatmap,
                image_batch,
                gt_hms,
            )
        ):
            plot = DetectionPlot(img)
            plot.draw_boxes(
                boxes=yolo_to_pascal(gt_boxes, (w, h)), labels=gt_labels, color="blue"
            )
            plot.draw_boxes(
                boxes=yolo_to_pascal(boxes, (w, h)),
                labels=labels,
                confidences=confidences,
                color="red",
            )
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")
            gt_merged_hm, _ = torch.max(gt_hm, dim=0)
            plot = DetectionPlot(gt_merged_hm * 255)
            plot.save(f"{self.out_dir}/{self.prefix}-gt-hm-{i}.png")
            merged_hm, _ = torch.max(hm, dim=0)
            plot = DetectionPlot(merged_hm * 255)
            plot.save(f"{self.out_dir}/{self.prefix}-hm-{i}.png")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        criterion: Criterion,
        optimizer: t.Any,
        visualize: Visualize,
        get_score: Callable[[PascalBoxes, PascalBoxes], float],
        to_boxes: ToBoxes,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.to_boxes = to_boxes
        self.model = model.to(self.device)
        self.get_score = get_score
        self.scaler = GradScaler()

        self.model_loader = model_loader
        self.visualize = visualize
        self.meters = {
            key: MeanMeter()
            for key in [
                "train_loss",
                "train_hm",
                "train_bm",
                "test_loss",
                "test_hm",
                "test_bm",
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
        for _ in range(epochs):
            self.train_one_epoch()
            self.eval_one_epoch()
            self.log()
            self.reset_meters()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(loader):
            gt_box_batch = [x.to(self.device) for x in gt_box_batch]
            gt_label_batch = [x.to(self.device) for x in gt_label_batch]
            image_batch = image_batch.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                netout = self.model(image_batch)
                loss, hm_loss, bm_loss, _ = self.criterion(
                    image_batch, netout, gt_box_batch, gt_label_batch
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.meters["train_loss"].update(loss.item())
            self.meters["train_hm"].update(hm_loss.item())
            self.meters["train_bm"].update(bm_loss.item())

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.eval()
        loader = self.test_loader
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(loader):
            image_batch = image_batch.to(self.device)
            gt_box_batch = [x.to(self.device) for x in gt_box_batch]
            gt_label_batch = [x.to(self.device) for x in gt_label_batch]
            _, _, h, w = image_batch.shape
            netout = self.model(image_batch)
            loss, hm_loss, bm_loss, gt_hms = self.criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = self.to_boxes(netout)
            for boxes, gt_boxes in zip(box_batch, gt_box_batch):
                self.meters["score"].update(
                    self.get_score(
                        yolo_to_pascal(boxes, (w, h)),
                        yolo_to_pascal(gt_boxes, (w, h)),
                    )
                )

            self.meters["test_loss"].update(loss.item())
            self.meters["test_hm"].update(hm_loss.item())
            self.meters["test_bm"].update(bm_loss.item())

        self.visualize(
            netout,
            box_batch,
            confidence_batch,
            label_batch,
            gt_box_batch,
            gt_label_batch,
            image_batch,
            gt_hms,
        )
        self.model_loader.save_if_needed(
            self.model,
            self.meters[self.model_loader.key].get_value(),
        )
