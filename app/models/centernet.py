import torch
import numpy as np
import typing as t
import torchvision
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from logging import getLogger
from .modules import ConvBR2d
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from app.dataset import Targets, Target
from scipy.stats import multivariate_normal
from .utils import box_cxcywh_to_xyxy
from app.utils import plot_heatmap, DetectionPlot
from pathlib import Path
from app.meters import EMAMeter
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

    def __call__(self, samples: NetInputs, src: NetOutputs, tgt: NetOutputs) -> None:
        src = {k: v[: self.limit] for k, v in src.items()}  # type: ignore
        tgt = {k: v[: self.limit] for k, v in tgt.items()}  # type: ignore
        src_boxes = self.to_boxes(src)
        images = samples["images"][: self.limit].detach().cpu()
        tgt_boxes = self.to_boxes(tgt)
        for i, (sb, tb, img) in enumerate(zip(src_boxes, tgt_boxes, images)):
            plot = DetectionPlot()
            plot.with_image(img)
            plot.with_boxes(tb[1], tb[0], color="blue")
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
        num_pos = pos_mask.sum().clamp(1)
        return loss / num_pos.float()


def gaussian_2d(shape: t.Any, sigma: float = 1) -> np.ndarray:
    m, n = int((shape[0] - 1.0) / 2.0), int((shape[1] - 1.0) / 2.0)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class Backbone(nn.Module):
    def __init__(self, name: str, out_channels: int) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        if name == "resnet34" or name == "resnet18":
            num_channels = 512
        else:
            num_channels = 2048
        self.layers = list(self.backbone.children())[:-2]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=i, out_channels=out_channels, kernel_size=1,)
                for i in [64, 64, 128, 256, 512]
            ]
        )

    def forward(self, x: Tensor) -> FP:
        internal_outputs = []
        for layer in self.layers:
            x = layer(x)
            internal_outputs.append(x)
        _, p3, _, p4, _, p5, p6, p7 = internal_outputs
        return (
            self.projects[0](p3),
            self.projects[1](p4),
            self.projects[2](p5),
            self.projects[3](p6),
            self.projects[4](p7),
        )


class HardHeatMap(nn.Module):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, boxes: Tensor) -> "NetOutputs":
        heatmap = torch.zeros((self.h, self.w), dtype=torch.float32).to(boxes.device)
        sizemap = torch.zeros((2, self.h, self.w), dtype=torch.float32).to(boxes.device)
        if len(boxes) == 0:
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
        mount = gaussian_2d((64, 64), sigma=7)
        self.mount = torch.tensor(mount, dtype=torch.float32).view(
            1, 1, mount.shape[0], mount.shape[1]
        )

    def forward(self, boxes: Tensor) -> Tensor:
        img = torch.zeros((1, 1, self.w, self.h), dtype=torch.float32).to(boxes.device)
        b, _ = boxes.shape
        if b == 0:
            return img
        xyxy = box_cxcywh_to_xyxy(boxes)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * self.w
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * self.w
        xyxy = xyxy.long()
        sizes = xyxy[:, [2, 3]] - xyxy[:, [0, 1]]
        for box, size in zip(xyxy, sizes):
            x, y = box[0], box[1]
            w, h = size[0], size[1]
            mount = F.interpolate(self.mount, size=(w, h)).to(boxes.device)
            mount = torch.max(mount, img[:, :, x : x + w, y : y + h])  # type: ignore
            img[:, :, x : x + w, y : y + h] = mount  # type: ignore
        return img


class PreProcess(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.h = 1024 // 2 ** config.scale_factor
        self.w = 1024 // 2 ** config.scale_factor
        self.heatmap = HardHeatMap(w=self.w, h=self.h)

    def forward(
        self, batch: t.Tuple[Tensor, t.List[Target]]
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


#  class Evaluate:
#      def __init__(self) -> None:
#          ...
#
#      def __call__(self, preds: NetOutputs, targets: NetInputs) -> None:
#          ...

#  class PostProcess:
#      def __init__(self) -> None:
#          self.to_boxes = ToBoxes(thresold=0.1)
#      def __call__(self, samples: NetInputs, preds: NetOutputs, targets: NetInputs) -> t.List[t.Tuple[Tensor, Tensor]]:
#          tgt_boxes = self.to_boxes(targets)
#          pred_boxes = self.to_boxes(preds)
#          imgs = samples.detach().cpu()
#          for i, (sb, tb, hm) in enuzip(pred_boxes, tgt_boxes, imgs):
#              plot = DetectionPlot()
#              plot.with_image(hm[0])
#              plot.with_boxes(sb[1], sb[0], color="red")
#              plot.with_boxes(tb[1], tb[0])
#              plot.save(str(self.output_dir.joinpath(f"{self.prefix}-boxes.png")))


class Criterion(nn.Module):
    def __init__(self, name: str = "train") -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.hm_meter = EMAMeter(f"{name}-hm")
        self.size_meter = EMAMeter(f"{name}-size")

    def forward(self, src: NetOutputs, tgt: NetOutputs) -> Tensor:
        # TODO test code
        b, _, _, _ = src["heatmap"].shape
        hm_loss = self.focal_loss(src["heatmap"], tgt["heatmap"]) / b
        size_loss = (
            (
                F.l1_loss(src["sizemap"], tgt["sizemap"], reduction="none")
                * tgt["heatmap"]
            ).sum()
            / b
            / 20
        )
        self.hm_meter.update(hm_loss.item())
        self.size_meter.update(size_loss.item())
        return hm_loss + size_loss


class Reg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(SENextBottleneck2d(in_channels, channels))

        self.out = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
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


class Augmention:
    def __call__(self, images: Tensor) -> None:
        transform = albm.Compose([albm.VerticalFlip(), albm.HorizontalFlip(),])


class CenterNet(nn.Module):
    def __init__(
        self, name: str = "resnet18", num_classes: int = 1, num_queries: int = 100
    ) -> None:
        super().__init__()
        channels = 64
        self.backbone = Backbone(name, out_channels=channels)
        self.fpn = nn.Sequential(BiFPN(channels=channels), BiFPN(channels=channels),)
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
