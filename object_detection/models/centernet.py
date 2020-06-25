import torch
import numpy as np
import typing as t
import math
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn, Tensor
from logging import getLogger
from object_detection.entities.box import (
    YoloBoxes,
    Confidences,
    yolo_to_pascal,
    PascalBoxes,
    pascal_to_yolo,
)
from object_detection.utils import DetectionPlot
from object_detection.entities.image import ImageId
from .modules import ConvBR2d
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .backbones import EfficientNetBackbone, ResNetBackbone
from object_detection.meters import BestWatcher, EMAMeter
from object_detection.entities import ImageBatch, PredBoxes, Image, Batch
from torchvision.ops import nms
from torch.utils.data import DataLoader
from object_detection.model_loader import ModelLoader

from pathlib import Path

logger = getLogger(__name__)


def collate_fn(batch: Batch) -> Tuple[ImageBatch, List[YoloBoxes], List[ImageId]]:
    images: List[t.Any] = []
    id_batch: List[ImageId] = []
    box_batch: List[YoloBoxes] = []

    for id, img, boxes in batch:
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
    return ImageBatch(torch.stack(images)), box_batch, id_batch


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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.conv(x)
        x = self.out(x)
        return x


Heatmap = t.NewType("Heatmap", Tensor)  # [B, 1, H, W]
Sizemap = t.NewType("Sizemap", Tensor)  # [B, 2, H, W]
NetOutput = Tuple[Heatmap, Sizemap]


class Up2d(nn.Module):
    up: t.Union[nn.Upsample, nn.ConvTranspose2d]

    def __init__(self, channels: int, bilinear: bool = False,) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x):  # type: ignore
        x = self.up(x)
        return x


class CenterNet(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.channels = channels
        self.backbone = EfficientNetBackbone(1, out_channels=channels)
        self.fpn = nn.Sequential(BiFPN(channels=channels))
        self.heatmap = nn.Sequential(
            Reg(in_channels=channels, out_channels=1, depth=1), nn.Sigmoid(),
        )
        self.box_size = nn.Sequential(
            Reg(in_channels=channels, out_channels=2, depth=1), nn.Sigmoid(),
        )

    def forward(self, x: ImageBatch) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmap = self.heatmap(fp[0])
        sizemap = self.box_size(fp[0])
        return Heatmap(heatmap), Sizemap(sizemap)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        alpha = self.alpha
        beta = self.beta
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        neg_loss = (
            -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
        )
        loss = (pos_loss + neg_loss).sum()
        num_pos = pos_mask.sum().float()
        return loss / num_pos


class Criterion:
    def __init__(
        self, heatmap_weight: float = 1.0, sizemap_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.reg_loss = RegLoss()
        self.sizemap_weight = sizemap_weight
        self.heatmap_weight = heatmap_weight

    def __call__(self, src: NetOutput, tgt: NetOutput) -> Tuple[Tensor, Tensor, Tensor]:
        s_hm, s_sm = src
        t_hm, t_sm = tgt
        hm_loss = self.focal_loss(s_hm, t_hm) * self.heatmap_weight
        sm_loss = self.reg_loss(s_sm, t_sm) * self.sizemap_weight
        loss = hm_loss + sm_loss
        return (
            loss,
            hm_loss,
            sm_loss,
        )


class RegLoss:
    def __call__(self, output: Sizemap, target: Sizemap,) -> Tensor:
        mask = (target > 0).view(target.shape)
        num = mask.sum()
        regr_loss = F.l1_loss(output, target, reduction="none") * mask
        regr_loss = regr_loss.sum() / (num + 1e-4)
        return regr_loss


def gaussian_2d(shape: t.Any, sigma: float = 1) -> np.ndarray:
    m, n = int((shape[0] - 1.0) / 2.0), int((shape[1] - 1.0) / 2.0)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class ToBoxes:
    def __init__(self, thresold: float, limit: int = 100) -> None:
        self.limit = limit
        self.thresold = thresold

    def __call__(self, inputs: NetOutput) -> t.List[t.Tuple[YoloBoxes, Confidences]]:
        heatmaps, sizemaps = inputs
        device = heatmaps.device
        kp_maps = (F.max_pool2d(heatmaps, 3, stride=1, padding=1) == heatmaps) & (
            heatmaps > self.thresold
        )
        batch_size, _, height, width = heatmaps.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        rows: t.List[t.Tuple[YoloBoxes, Confidences]] = []
        for hm, kp_map, size_map in zip(
            heatmaps.squeeze(1), kp_maps.squeeze(1), sizemaps
        ):
            pos = kp_map.nonzero()
            confidences = hm[pos[:, 0], pos[:, 1]]
            wh = size_map[:, pos[:, 0], pos[:, 1]]
            cxcy = pos[:, [1, 0]].float() / original_wh
            boxes = torch.cat([cxcy, wh.permute(1, 0)], dim=1)
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            rows.append(
                (YoloBoxes(boxes[sort_idx]), Confidences(confidences[sort_idx]))
            )
        return rows


class SoftHeatMap:
    def __init__(
        self, mount_size: t.Tuple[int, int] = (5, 5), sigma: float = 1,
    ) -> None:
        self.mount_size = mount_size
        self.mount_pad = (
            self.mount_size[0] % 2,
            self.mount_size[1] % 2,
        )
        mount = gaussian_2d(self.mount_size, sigma=sigma)
        self.mount = torch.tensor(mount, dtype=torch.float32).view(
            1, 1, mount.shape[0], mount.shape[1]
        )
        self.mount = self.mount / self.mount.max()

    def __call__(self, boxes: YoloBoxes, size: t.Tuple[int, int]) -> NetOutput:
        device = boxes.device
        w, h = size
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        sizemap = torch.zeros((1, 2, h, w), dtype=torch.float32).to(device)
        box_count, _ = boxes.shape
        if box_count == 0:
            return Heatmap(heatmap), Sizemap(sizemap)
        box_cx, box_cy, box_w, box_h = torch.unbind(boxes, dim=1)
        box_cx = (box_cx * w).long()
        box_cy = (box_cy * h).long()
        sizemap[:, :, box_cy, box_cx] = torch.stack([box_w, box_h], dim=0)
        mount_w, mount_h = self.mount_size
        pad_w, pad_h = self.mount_pad
        mount_x0 = box_cx - mount_h // 2
        mount_x1 = box_cx + mount_h // 2 + pad_h
        mount_y0 = box_cy - mount_w // 2
        mount_y1 = box_cy + mount_w // 2 + pad_w

        mount = self.mount.to(device)
        for x0, x1, y0, y1 in zip(mount_x0, mount_x1, mount_y0, mount_y1):
            target = heatmap[:, :, y0:y1, x0:x1]  # type: ignore
            _, _, target_h, target_w = target.shape
            if (target_h >= mount_h) and (target_w >= mount_w):
                mount = torch.max(mount, target)
                heatmap[:, :, y0:y1, x0:x1] = mount  # type: ignore
        return Heatmap(heatmap), Sizemap(sizemap)


class PreProcess:
    def __init__(self, device: t.Any) -> None:
        super().__init__()
        self.heatmap = SoftHeatMap()
        self.device = device

    def __call__(
        self, batch: t.Tuple[ImageBatch, t.List[YoloBoxes]]
    ) -> t.Tuple[ImageBatch, NetOutput]:
        image_batch, boxes_batch = batch
        image_batch = ImageBatch(image_batch.to(self.device))
        hms: t.List[t.Any] = []
        sms: t.List[t.Any] = []
        _, _, h, w = image_batch.shape
        for img, boxes in zip(image_batch.unbind(0), boxes_batch):
            hm, sm = self.heatmap(YoloBoxes(boxes.to(self.device)), (w // 2, h // 2))
            hms.append(hm)
            sms.append(sm)

        heatmap = torch.cat(hms, dim=0)
        sizemap = torch.cat(sms, dim=0)
        return image_batch, (Heatmap(heatmap), Sizemap(sizemap))


class PostProcess:
    def __init__(self) -> None:
        self.to_boxes = ToBoxes(thresold=0.1, limit=300)

    def __call__(
        self, x: NetOutput, image_ids: List[ImageId], images: ImageBatch
    ) -> List[Tuple[ImageId, YoloBoxes, Confidences]]:
        _, _, h, w = images.shape
        rows = []
        for image_id, (boxes, confidences) in zip(image_ids, self.to_boxes(x)):
            rows.append((image_id, boxes, confidences))
        return rows


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
        net_out: NetOutput,
        src: List[Tuple[ImageId, YoloBoxes, Confidences]],
        tgt: List[YoloBoxes],
        image_batch: ImageBatch,
    ) -> None:
        heatmap, _ = net_out
        image_batch = ImageBatch(image_batch[: self.limit])
        heatmap = Heatmap(heatmap[: self.limit])
        src = src[: self.limit]
        tgt = tgt[: self.limit]
        _, _, h, w = image_batch.shape
        for i, ((_, sb, sc), tb, hm, img) in enumerate(
            zip(src, tgt, heatmap, image_batch)
        ):
            plot = DetectionPlot(h=h, w=w, use_alpha=self.use_alpha)
            plot.with_image(img, alpha=0.5)
            plot.with_image(hm[0], alpha=0.5)
            plot.with_yolo_boxes(tb, color="blue")
            plot.with_yolo_boxes(sb, sc, color="red")
            plot.save(f"{self.out_dir}/{self.prefix}-boxes-{i}.png")


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        optimizer: t.Any,
        visualize: Visualize,
        device: str = "cpu",
        criterion: Criterion = Criterion(),
    ) -> None:
        self.device = torch.device(device)
        self.model_loader = model_loader
        self.model = model_loader.model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.preprocess = PreProcess(self.device)
        self.post_process = PostProcess()
        self.best_watcher = BestWatcher()
        self.visualize = visualize
        self.meters = {
            key: EMAMeter(key)
            for key in [
                "test_loss",
                "test_hm",
                "test_sm",
                "train_loss",
                "train_sm",
                "train_hm",
            ]
        }

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()
            self.eval_one_epoch()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for samples, targets, ids in loader:
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss, hm_loss, sm_loss = self.criterion(outputs, cri_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.meters["train_loss"].update(loss.item())
            self.meters["train_hm"].update(hm_loss.item())
            self.meters["train_sm"].update(sm_loss.item())

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.eval()
        loader = self.test_loader
        for samples, targets, ids in loader:
            samples, cri_targets = self.preprocess((samples, targets))
            outputs = self.model(samples)
            loss, hm_loss, sm_loss = self.criterion(outputs, cri_targets)
            preds = self.post_process(outputs, ids, samples)

            self.meters["test_loss"].update(loss.item())
            self.meters["test_hm"].update(hm_loss.item())
            self.meters["test_sm"].update(sm_loss.item())

        if self.best_watcher.step(self.meters["test_loss"].get_value()):
            self.model_loader.save({"loss": self.meters["test_loss"].get_value()})
        self.visualize(outputs, preds, targets, samples)
