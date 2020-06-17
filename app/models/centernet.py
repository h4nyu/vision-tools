import torch
import numpy as np
import typing as t
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from logging import getLogger
from .modules import ConvBR2d
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .losses import BoxIoU
from app.dataset import Targets, Target
from scipy.stats import multivariate_normal
from .utils import box_cxcywh_to_xyxy, box_iou
from .backbones import EfficientNetBackbone, ResNetBackbone
from app.utils import plot_heatmap, DetectionPlot
from pathlib import Path
from app.meters import EMAMeter
from app.eval import MeamPrecition
import albumentations as albm

logger = getLogger(__name__)

Outputs = t.TypedDict(
    "Outputs", {"heatmap": Tensor, "box_size": Tensor,}  # [B, 1, W, H]  # [B, 2, W, H]
)

NetInputs = t.TypedDict("NetInputs", {"images": Tensor})

NetOutputs = t.TypedDict("NetOutputs", {"heatmap": Tensor, "sizemap": Tensor,})


class VisualizeHeatmap:
    def __init__(self, output_dir: Path, prefix: str = "", limit: int = 1) -> None:
        self.prefix = prefix
        self.output_dir = output_dir
        self.limit = limit
        self.to_boxes = ToBoxes(thresold=0.1, limit=50)

    def __call__(
        self,
        samples: NetInputs,
        src: NetOutputs,
        cri_targets: NetOutputs,
        targets: Targets,
    ) -> None:
        src = {k: v[: self.limit] for k, v in src.items()}  # type: ignore
        src_boxes = self.to_boxes(src)
        tgt_boxes = [t["boxes"] for t in targets][: self.limit]
        heatmaps = src["heatmap"][: self.limit].detach().cpu()
        images = samples["images"][: self.limit].detach().cpu()
        for i, (sb, tb, img, hm) in enumerate(
            zip(src_boxes, tgt_boxes, images, heatmaps)
        ):
            plot = DetectionPlot()
            plot.with_image(hm[0])
            plot.with_boxes(tb, color="blue")
            plot.with_boxes(sb[1], sb[0], color="red")
            plot.save(str(self.output_dir.joinpath(f"{self.prefix}-boxes-{i}.png")))


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


def gaussian_2d(shape: t.Any, sigma: float = 1) -> np.ndarray:
    m, n = int((shape[0] - 1.0) / 2.0), int((shape[1] - 1.0) / 2.0)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class HardHeatMap(nn.Module):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, boxes: Tensor) -> "NetOutputs":
        heatmap = torch.zeros((self.h, self.w), dtype=torch.float32).to(boxes.device)
        sizemap = torch.zeros((2, self.h, self.w), dtype=torch.float32).to(boxes.device)
        if len(boxes) == 0:
            sizemap = sizemap.unsqueeze(0)
            return dict(heatmap=heatmap, sizemap=sizemap)
        cx, cy = (boxes[:, 0] * self.w).long(), (boxes[:, 1] * self.h).long()
        heatmap[cy, cx] = 1.0
        sizemap[:, cy, cx] = boxes[:, 2:4].permute(1, 0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        sizemap = sizemap.unsqueeze(0)
        return dict(heatmap=heatmap, sizemap=sizemap)


class SoftHeatMap(nn.Module):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h
        self.mount_size = (3, 3)
        self.mount_pad = (
            self.mount_size[0] % 2,
            self.mount_size[1] % 2,
        )
        mount = gaussian_2d(self.mount_size, sigma=1)
        self.mount = torch.tensor(mount, dtype=torch.float32).view(
            1, 1, mount.shape[0], mount.shape[1]
        )
        self.mount = self.mount / self.mount.max()

    def forward(self, boxes: Tensor) -> "NetOutputs":
        img = torch.zeros((1, 1, self.h, self.w), dtype=torch.float32).to(boxes.device)
        sizemap = torch.zeros((2, self.h, self.w), dtype=torch.float32).to(boxes.device)
        device = img.device
        b, _ = boxes.shape
        if b == 0:
            sizemap = sizemap.unsqueeze(0)
            return dict(heatmap=img, sizemap=sizemap)
        cx, cy = (boxes[:, 0] * self.w).long(), (boxes[:, 1] * self.h).long()
        sizemap[:, cy, cx] = boxes[:, 2:4].permute(1, 0)
        mount_w, mount_h = self.mount_size
        pad_w, pad_h = self.mount_pad
        mount_x0 = cx - mount_h // 2
        mount_x1 = cx + mount_h // 2 + pad_h
        mount_y0 = cy - mount_w // 2
        mount_y1 = cy + mount_w // 2 + pad_w

        for x0, x1, y0, y1 in zip(mount_x0, mount_x1, mount_y0, mount_y1):
            target = img[:, :, y0: y1,x0: x1] # type: ignore
            _, _, target_h, target_w = target.shape
            if (target_h >= mount_h) and (target_w >= mount_w):
                mount = torch.max(self.mount.to(device), target)
                img[:, :, y0 :  y1, x0 : x1] = mount  # type: ignore
        sizemap = sizemap.unsqueeze(0)
        return dict(heatmap=img, sizemap=sizemap)


class PreProcess(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.h = 1024 // 2 ** config.scale_factor
        self.w = 1024 // 2 ** config.scale_factor
        self.heatmap = SoftHeatMap(w=self.w, h=self.h)

    def forward(
        self, batch: t.Tuple[Tensor, Targets]
    ) -> t.Tuple[NetInputs, NetOutputs]:
        images, targets = batch

        hms = []
        sms = []
        for t in targets:
            boxes = t["boxes"]
            res = self.heatmap(t["boxes"])
            hms.append(res["heatmap"])
            sms.append(res["sizemap"])

        heatmap = torch.cat(hms, dim=0)
        sizemap = torch.cat(sms, dim=0)
        images = F.interpolate(images, size=(self.w * 2, self.h * 2))
        return dict(images=images), dict(heatmap=heatmap, sizemap=sizemap)


class Evaluate:
    def __init__(self) -> None:
        self.to_boxes = ToBoxes(thresold=0.1)
        self.mean_precision = MeamPrecition()

    def __call__(self, inputs: NetOutputs, targets: Targets) -> float:
        src_boxes = self.to_boxes(inputs)
        lenght = len(targets)
        score = 0.0
        for (_, sb), tgt in zip(src_boxes, targets):
            tb = tgt["boxes"]
            pred_boxes = box_cxcywh_to_xyxy(sb)
            gt_boxes = box_cxcywh_to_xyxy(tgt["boxes"])
            score += self.mean_precision(pred_boxes, gt_boxes)
        return score / lenght



class Criterion(nn.Module):
    def __init__(self, name: str = "train") -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.reg_loss = RegLoss()
        self.hm_meter = EMAMeter(f"{name}-hm")
        self.size_meter = EMAMeter(f"{name}-size")

    def forward(self, src: NetOutputs, tgt: NetOutputs) -> Tensor:
        # TODO test code
        b, _, _, _ = src["heatmap"].shape
        hm_loss = self.focal_loss(src["heatmap"], tgt["heatmap"]) / b
        size_loss = self.reg_loss(src['sizemap'], tgt['sizemap'], tgt["heatmap"]) / b
        self.hm_meter.update(hm_loss.item())
        self.size_meter.update(size_loss.item())
        return hm_loss + size_loss


class Reg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(SENextBottleneck2d(in_channels, in_channels))

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.conv(x)
        x = self.out(x)
        return x


class ToBoxes:
    def __init__(self, thresold: float, limit: int = 200) -> None:
        self.limit = limit
        self.thresold = thresold

    def __call__(self, inputs: NetOutputs) -> t.List[t.Tuple[Tensor, Tensor]]:
        """
        cxcywh 0-1
        """
        heatmaps = inputs["heatmap"]
        sizemaps = inputs["sizemap"]
        device = heatmaps.device
        kp_maps = (F.max_pool2d(heatmaps, 3, stride=1, padding=1) == heatmaps) & (
            heatmaps > self.thresold
        )
        batch_size, _, height, width = heatmaps.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        targets: t.List[t.Tuple[Tensor, Tensor]] = []
        for hm, kp_map, size_map in zip(
            heatmaps.squeeze(1), kp_maps.squeeze(1), sizemaps
        ):
            pos = kp_map.nonzero()
            confidences = hm[pos[:, 0], pos[:, 1]]
            wh = size_map[:, pos[:, 0], pos[:, 1]]
            cxcy = pos[:, [1, 0]].float() / original_wh
            boxes = torch.cat([cxcy, wh.permute(1, 0)], dim=1)
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            targets.append((confidences[sort_idx], boxes[sort_idx],))
        return targets


class RegLoss(nn.Module):
    def forward(self, output:Tensor, target:Tensor, mask:Tensor) -> Tensor:
        num = mask.eq(1).float().sum()
        mask = mask.expand_as(target).float()
        regr = output * mask
        gt_regr = target * mask
        regr_loss = F.l1_loss(regr, gt_regr, reduction="none").sum()
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


class CenterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 64
        self.backbone = EfficientNetBackbone(1, out_channels=channels)
        self.fpn = nn.Sequential(BiFPN(channels=channels))
        self.heatmap = Reg(in_channels=channels, out_channels=1)
        self.box_size = Reg(in_channels=channels, out_channels=2)

    def forward(self, inputs: NetInputs) -> NetOutputs:
        """
        x: [B, 3, W, H]
        """
        x = inputs["images"]
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmap = self.heatmap(fp[0])
        sizemap = self.box_size(fp[0])
        return dict(heatmap=heatmap, sizemap=sizemap)