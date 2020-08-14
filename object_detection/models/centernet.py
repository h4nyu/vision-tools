import torch
import numpy as np
import math
import typing as t
import torch as tr
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, NewType, Union, Callable, Any
from torch import nn, Tensor
from typing_extensions import Literal
from logging import getLogger
from tqdm import tqdm
from object_detection.entities import (
    BoxMaps,
    YoloBoxes,
    Confidences,
    PascalBoxes,
    PyramidIdx,
    yolo_to_pascal,
    pascal_to_yolo,
    yolo_to_coco,
    Labels,
)
from object_detection.utils import DetectionPlot
from object_detection.entities.image import ImageId
from .mkmaps import Heatmaps, MkMapsFn, MkBoxMapsFn
from .modules import ConvBR2d, Mish
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import Reduction
from object_detection.meters import MeanMeter
from object_detection.entities import ImageBatch, PredBoxes, Image, TrainSample
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
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return id_batch, ImageBatch(torch.stack(images)), box_batch, label_batch


class Reg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(
            *[SENextBottleneck2d(in_channels, in_channels) for _ in range(depth)]
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.out(x)
        return x


class CountReg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        channels = in_channels
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=in_channels, kernel_size=1),
            Mish(),
            nn.Conv2d(in_channels=channels, out_channels=in_channels, kernel_size=1),
            Mish(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.dense(x)
        x = self.out(x)
        x = x.view(-1)
        return x


Counts = NewType("Counts", Tensor)  # [B]
NetOutput = Tuple[Heatmaps, BoxMaps, Counts]  # label, pos, size, count


class CenterNet(nn.Module):
    def __init__(
        self,
        channels: int,
        backbone: nn.Module,
        depth: int = 2,
        out_idx: PyramidIdx = 4,
    ) -> None:
        super().__init__()
        self.out_idx = out_idx - 3
        self.channels = channels
        self.backbone = backbone
        self.fpn = nn.Sequential(*[BiFPN(channels=channels) for _ in range(depth)])
        self.hm_reg = nn.Sequential(
            Reg(in_channels=channels, out_channels=1, depth=depth), nn.Sigmoid(),
        )
        self.box_reg = nn.Sequential(
            Reg(in_channels=channels, out_channels=4, depth=depth),
        )

        self.count_reg = nn.Sequential(
            CountReg(in_channels=channels, out_channels=1, depth=depth),
        )

    def forward(self, x: ImageBatch) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmaps = Heatmaps(self.hm_reg(fp[self.out_idx]))
        boxmaps = self.box_reg(fp[self.out_idx])
        counts = self.count_reg(fp[-1])
        return heatmaps, BoxMaps(boxmaps), Counts(counts)


class HMLoss(nn.Module):
    """
    Modified focal loss
    """

    def __init__(
        self, alpha: float = 2.0, beta: float = 2.0, eps: float = 1e-4,
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
        self.reg_loss = RegLoss()
        self.heatmap_weight = heatmap_weight
        self.box_weight = box_weight
        self.count_weight = count_weight
        self.mk_hmmaps = mk_hmmaps
        self.mk_boxmaps = mk_boxmaps

    def __call__(
        self, images: ImageBatch, netout: NetOutput, gt_boxes: List[YoloBoxes]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Heatmaps]:
        s_hm, s_bm, s_c = netout
        _, _, orig_h, orig_w = images.shape
        _, _, h, w = s_hm.shape
        t_hm = self.mk_hmmaps(gt_boxes, (h, w), (orig_h, orig_w))
        t_bm = self.mk_boxmaps(gt_boxes, t_hm)
        hm_loss = self.hmloss(s_hm, t_hm) * self.heatmap_weight
        box_loss = self.reg_loss(s_bm, t_bm) * self.box_weight
        c_loss = torch.tensor(0.0)
        dm_loss = torch.tensor(0.0)
        loss = hm_loss + box_loss + c_loss
        return (loss, hm_loss, box_loss, dm_loss, c_loss, t_hm)


class RegLoss:
    def __call__(self, output: BoxMaps, target: BoxMaps) -> Tensor:
        mask = (target > 0).view(target.shape)
        num = mask.sum()
        regr_loss = F.l1_loss(output, target, reduction="none") * mask
        regr_loss = regr_loss.sum() / num.clamp(min=1.0)
        return regr_loss


class ToBoxes:
    def __init__(
        self,
        threshold: float = 0.1,
        kernel_size: int = 3,
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
    def __call__(self, inputs: NetOutput) -> t.List[t.Tuple[YoloBoxes, Confidences]]:
        heatmaps, boxmaps, counts = inputs
        device = heatmaps.device
        kpmap = (self.max_pool(heatmaps) == heatmaps) & (heatmaps > self.threshold)
        batch_size, _, height, width = heatmaps.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        rows: t.List[t.Tuple[YoloBoxes, Confidences]] = []
        for hm, km, bm in zip(
            heatmaps.squeeze(1), kpmap.squeeze(1), boxmaps
        ):
            kp = torch.nonzero(km, as_tuple=False)
            confidences = hm[kp[:, 0], kp[:, 1]]
            boxes = bm[:, kp[:, 0], kp[:, 1]].t()
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            rows.append(
                (YoloBoxes(boxes[sort_idx]), Confidences(confidences[sort_idx]))
            )
        return rows


class PreProcess:
    def __init__(self, device: t.Any) -> None:
        super().__init__()
        self.device = device

    def __call__(
        self, batch: t.Tuple[ImageBatch, t.List[YoloBoxes]]
    ) -> t.Tuple[ImageBatch, List[YoloBoxes]]:
        image_batch, box_batch = batch
        image_batch = ImageBatch(image_batch.to(self.device))
        box_batch = [YoloBoxes(x.to(self.device)) for x in box_batch]

        return image_batch, box_batch


class PostProcess:
    def __init__(self, to_boxes: ToBoxes = ToBoxes()) -> None:
        self.to_boxes = to_boxes

    def __call__(self, netout: NetOutput,) -> Tuple[List[YoloBoxes], List[Confidences]]:
        box_batch = []
        confidence_batch = []
        for boxes, confidences in self.to_boxes(netout):
            box_batch.append(boxes)
            confidence_batch.append(confidences)
        return box_batch, confidence_batch


class Visualize:
    def __init__(
        self,
        out_dir: str,
        prefix: str,
        limit: int = 1,
        use_alpha: bool = True,
        show_probs: bool = True,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        self.prefix = prefix
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.use_alpha = use_alpha
        self.show_probs = show_probs
        self.figsize = figsize

    @torch.no_grad()
    def __call__(
        self,
        net_out: NetOutput,
        src: Tuple[List[YoloBoxes], List[Confidences]],
        tgt: List[YoloBoxes],
        image_batch: ImageBatch,
        gt_hms: Heatmaps,
    ) -> None:
        heatmap, _, counts = net_out
        box_batch, confidence_batch = src
        box_batch = box_batch[: self.limit]
        confidence_batch = confidence_batch[: self.limit]
        _, _, h, w = image_batch.shape
        for i, (sb, sc, tb, hm, img, gt_hm, count) in enumerate(
            zip(box_batch, confidence_batch, tgt, heatmap, image_batch, gt_hms, counts)
        ):
            plot = DetectionPlot(
                h=h,
                w=w,
                use_alpha=self.use_alpha,
                figsize=self.figsize,
                show_probs=self.show_probs,
            )
            plot.set_title(f"count: {count:.4f}")
            plot.with_image(img, alpha=0.7)
            plot.with_image(hm[0].log(), alpha=0.3)
            plot.with_yolo_boxes(tb, color="blue")
            plot.with_yolo_boxes(sb, sc, color="red")
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")

            plot = DetectionPlot(
                h=h, w=w, use_alpha=self.use_alpha, figsize=self.figsize
            )
            plot.with_image(img, alpha=0.7)
            plot.with_image((gt_hm[0] + 1e-4).log(), alpha=0.3)
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
        get_score: Callable[[YoloBoxes, YoloBoxes], float],
        to_boxes: ToBoxes,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.preprocess = PreProcess(self.device)
        self.post_process = PostProcess(to_boxes=to_boxes)
        self.model = model.to(self.device)
        self.get_score = get_score

        self.model_loader = model_loader
        self.visualize = visualize
        self.meters = {
            key: MeanMeter()
            for key in [
                "train_loss",
                "train_sm",
                "train_hm",
                "train_dm",
                "train_c",
                "test_loss",
                "test_hm",
                "test_sm",
                "test_dm",
                "test_c",
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
        for ids, images, boxes, labels in tqdm(loader):
            images, boxes = self.preprocess((images, boxes))
            outputs = self.model(images)
            loss, hm_loss, sm_loss, dm_loss, c_loss, _ = self.criterion(
                images, outputs, boxes
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.meters["train_loss"].update(loss.item())
            self.meters["train_hm"].update(hm_loss.item())
            self.meters["train_sm"].update(sm_loss.item())
            self.meters["train_dm"].update(dm_loss.item())
            self.meters["train_c"].update(c_loss.item())

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.eval()
        loader = self.test_loader
        for ids, images, boxes, labels in tqdm(loader):
            images, boxes = self.preprocess((images, boxes))
            outputs = self.model(images)
            loss, hm_loss, sm_loss, dm_loss, c_loss, gt_hms = self.criterion(
                images, outputs, boxes
            )
            preds = self.post_process(outputs)
            for (pred, gt) in zip(preds[0], boxes):
                self.meters["score"].update(self.get_score(pred, gt))

            self.meters["test_loss"].update(loss.item())
            self.meters["test_hm"].update(hm_loss.item())
            self.meters["test_sm"].update(sm_loss.item())
            self.meters["test_dm"].update(dm_loss.item())
            self.meters["test_c"].update(c_loss.item())

        self.visualize(outputs, preds, boxes, images, gt_hms)
        self.model_loader.save_if_needed(
            self.model, self.meters[self.model_loader.key].get_value(),
        )
