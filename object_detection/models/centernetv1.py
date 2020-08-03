import torch
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor
from typing import Tuple, List, NewType, Callable, Any
from torch.utils.data import DataLoader
from typing_extensions import Literal
from pathlib import Path
from object_detection.meters import MeanMeter
from object_detection.utils import DetectionPlot
from .bottlenecks import SENextBottleneck2d
from .centernet import (
    Sizemap,
    DiffMap,
    HMLoss,
    PreProcess,
    Labels,
    ImageId,
    Heatmaps,
)
from logging import getLogger
from tqdm import tqdm
from .efficientdet import RegressionModel
from .bifpn import BiFPN
from torchvision.ops.boxes import box_iou
from torchvision.ops import nms
from object_detection.model_loader import ModelLoader
from object_detection.entities import (
    BoxMap,
    BoxMaps,
    PyramidIdx,
    ImageBatch,
    YoloBoxes,
    Confidences,
    yolo_to_pascal,
    pascal_to_yolo,
    boxmap_to_boxes,
    yolo_hflip,
    yolo_vflip,
    PascalBoxes,
    TrainSample,
    PredictionSample,
    yolo_clamp,
)
from .tta import HFlipTTA, VFlipTTA


logger = getLogger(__name__)


Heatmap = NewType("Heatmap", Tensor)  # [1, H, W]
NetOutput = Tuple[BoxMap, BoxMaps, Heatmaps]


def collate_fn(
    batch: List[TrainSample],
) -> Tuple[ImageBatch, List[YoloBoxes], List[Labels], List[ImageId]]:
    images: List[Any] = []
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


class Anchors:
    def __init__(self, size: int = 1) -> None:
        self.size = size

    def __call__(self, ref_images: Tensor) -> Any:
        h, w = ref_images.shape[-2:]
        device = ref_images.device
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.float32) / h,
            torch.arange(w, dtype=torch.float32) / w,
        )
        box_h = torch.ones((h, w)) * (self.size / h)
        box_w = torch.ones((h, w)) * (self.size / w)
        anchors = torch.stack([grid_x, grid_y, box_w, box_h])
        return anchors.to(device)


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
        preds: Tuple[List[YoloBoxes], List[Confidences]],
        gts: List[YoloBoxes],
        image_batch: ImageBatch,
        gt_hms: Heatmaps,
    ) -> None:
        _, _, heatmaps = net_out
        box_batch, confidence_batch = preds
        box_batch = box_batch[: self.limit]
        confidence_batch = confidence_batch[: self.limit]
        _, _, h, w = image_batch.shape
        for i, (sb, sc, tb, hm, img, gt_hm) in enumerate(
            zip(box_batch, confidence_batch, gts, heatmaps, image_batch, gt_hms)
        ):
            plot = DetectionPlot(
                h=h,
                w=w,
                use_alpha=self.use_alpha,
                figsize=self.figsize,
                show_probs=self.show_probs,
            )
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


class CenterNetV1(nn.Module):
    def __init__(
        self,
        channels: int,
        backbone: nn.Module,
        out_idx: PyramidIdx = 4,
        fpn_depth: int = 1,
        box_depth: int = 1,
        hm_depth: int = 1,
        anchors: Anchors = Anchors(),
    ) -> None:
        super().__init__()
        self.out_idx = out_idx - 3
        self.channels = channels
        self.backbone = backbone
        self.fpn = nn.Sequential(*[BiFPN(channels=channels) for _ in range(fpn_depth)])
        self.hm_reg = nn.Sequential(
            Reg(in_channels=channels, out_channels=channels, depth=hm_depth)
        )
        self.hm_out = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.anchors = anchors
        self.box_reg = nn.Sequential(
            Reg(in_channels=channels, out_channels=channels, depth=box_depth),
        )
        self.box_out = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: ImageBatch) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        h_fp = self.hm_reg(fp[self.out_idx])
        heatmaps = self.hm_out(h_fp)
        anchors = self.anchors(heatmaps)
        diffmaps = self.box_out(self.box_reg(fp[self.out_idx]))
        return anchors, BoxMaps(diffmaps), Heatmaps(heatmaps)


class ToBoxes:
    def __init__(
        self, threshold: float = 0.1, kernel_size: int = 5, use_peak: bool = False,
    ) -> None:
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.use_peak = use_peak
        self.max_pool = partial(
            F.max_pool2d, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        )

    @torch.no_grad()
    def __call__(self, inputs: NetOutput) -> Tuple[List[YoloBoxes], List[Confidences]]:
        anchormap, box_diffs, heatmap = inputs
        device = heatmap.device
        if self.use_peak:
            kpmap = (self.max_pool(heatmap) == heatmap) & (heatmap > self.threshold)
        else:
            kpmap = heatmap > self.threshold
        batch_size, _, height, width = heatmap.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        rows: List[Tuple[YoloBoxes, Confidences]] = []
        box_batch = []
        conf_batch = []
        for hm, km, box_diff in zip(heatmap.squeeze(1), kpmap.squeeze(1), box_diffs):
            kp = torch.nonzero(km, as_tuple=False)
            confidences = hm[kp[:, 0], kp[:, 1]]
            anchor = anchormap[:, kp[:, 0], kp[:, 1]].t()
            box_diff = box_diff[:, kp[:, 0], kp[:, 1]].t()
            boxes = anchor + box_diff
            box_batch.append(yolo_clamp(YoloBoxes(boxes)))
            conf_batch.append(Confidences(confidences))
        return box_batch, conf_batch


class NearnestAssign:
    def __call__(self, pred: YoloBoxes, gt: YoloBoxes) -> Tuple[Tensor, Tensor]:
        pred_count = pred.shape[0]
        gt_count = gt.shape[0]
        pred_ctr = pred[:, :2].view(pred_count, 1, 2).expand(pred_count, gt_count, 2,)
        gt_ctr = gt[:, :2]
        matrix = ((pred_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        min_dist, matched_idx = matrix.min(dim=1)
        max_lenght = (gt[:, 2:] / 2).min(dim=1)[0][matched_idx]
        filter_idx = min_dist < max_lenght
        return matched_idx, filter_idx


class BoxLoss:
    def __init__(self) -> None:
        self.assign = NearnestAssign()

    def __call__(
        self, anchormap: BoxMap, diffmap: BoxMap, gt_boxes: YoloBoxes, heatmap: Heatmap,
    ) -> Tensor:
        device = diffmap.device
        box_diff = boxmap_to_boxes(diffmap)
        anchors = boxmap_to_boxes(anchormap)
        centers = anchors[:, 2:]
        gt_centers = gt_boxes[:, 2:]

        match_indices, positive_indices = self.assign(anchors, gt_boxes,)
        num_pos = positive_indices.sum()
        if num_pos == 0:
            return torch.tensor(0.0).to(device)
        matched_gt_boxes = gt_boxes[match_indices][positive_indices]
        matched_anchors = anchors[positive_indices]
        pred_diff = box_diff[positive_indices]
        gt_diff = matched_gt_boxes - matched_anchors
        return F.l1_loss(pred_diff, gt_diff, reduction="mean")


MkMapsFn = Callable[[List[YoloBoxes], Tuple[int, int], Tuple[int, int],], Heatmaps]


class MkMapsBase:
    def _mkmaps(
        self, boxes: YoloBoxes, hw: Tuple[int, int], original_hw: Tuple[int, int]
    ) -> Heatmaps:
        ...

    @torch.no_grad()
    def __call__(
        self,
        box_batch: List[YoloBoxes],
        hw: Tuple[int, int],
        original_hw: Tuple[int, int],
    ) -> Heatmaps:
        hms: List[Tensor] = []
        for boxes in box_batch:
            hm = self._mkmaps(boxes, hw, original_hw)
            hms.append(hm)
        return Heatmaps(torch.cat(hms, dim=0))


class MkCrossMaps(MkMapsBase):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def _mkmaps(
        self, boxes: YoloBoxes, hw: Tuple[int, int], original_hw: Tuple[int, int]
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)
        step_w = 1.0 / w
        step_h = 1.0 / h
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(start=0.0, end=1.0, step=step_h, dtype=torch.float32).to(
                device
            ),
            torch.arange(start=0.0, end=1.0, step=step_w, dtype=torch.float32).to(
                device
            ),
        )
        #  img_wh = torch.tensor([w, h]).to(device)
        grid_cxcy = (boxes[:, :2]).view(box_count, 2, 1, 1).expand((box_count, 2, h, w))
        grid_wh = (boxes[:, 2:]).view(box_count, 2, 1, 1).expand((box_count, 2, h, w))
        grid_xy = torch.stack([grid_x, grid_y]).expand((box_count, 2, h, w))
        grid_xycxcywh = torch.cat([grid_cxcy, grid_xy, grid_wh], dim=1)
        mounts = (
            (grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]).abs() < step_w
        ) & (((grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]).abs() < step_h))
        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                - grid_xycxcywh[:, 4, ::] * self.alpha / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w / 2
        ) & (((grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]).abs() < step_h / 2))
        mounts |= (
            (
                grid_xycxcywh[:, 2, ::]
                + grid_xycxcywh[:, 4, ::] * self.alpha / 2
                - grid_xycxcywh[:, 0, ::]
            ).abs()
            < step_w / 2
        ) & (((grid_xycxcywh[:, 3, ::] - grid_xycxcywh[:, 1, ::]).abs() < step_h / 2))
        mounts |= (
            (
                grid_xycxcywh[:, 3, ::]
                - grid_xycxcywh[:, 5, ::] * self.alpha / 2
                - grid_xycxcywh[:, 1, ::]
            ).abs()
            < step_w / 2
        ) & (((grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]).abs() < step_w / 2))
        mounts |= (
            (
                grid_xycxcywh[:, 3, ::]
                + grid_xycxcywh[:, 5, ::] * self.alpha / 2
                - grid_xycxcywh[:, 1, ::]
            ).abs()
            < step_w / 2
        ) & (((grid_xycxcywh[:, 2, ::] - grid_xycxcywh[:, 0, ::]).abs() < step_w / 2))
        mounts = mounts.view(box_count, 1, h, w).float()
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


GaussianMapMode = Literal["length", "aspect", "constant"]


class MkGaussianMaps(MkMapsBase):
    def __init__(self, sigma: float = 0.5, mode: GaussianMapMode = "length",) -> None:
        self.sigma = sigma
        self.mode = mode

    def _mkmaps(
        self, boxes: YoloBoxes, hw: Tuple[int, int], original_hw: Tuple[int, int]
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64), torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        if self.mode == "aspect":
            weight = (boxes[:, 2:] ** 2).clamp(min=1e-4).view(box_count, 2, 1, 1)
        elif self.mode == "length":
            weight = (
                (boxes[:, 2:] ** 2)
                .min(dim=1, keepdim=True)[0]
                .clamp(min=1e-4)
                .view(box_count, 1, 1, 1)
            )
        else:
            weight = torch.ones((box_count, 1, 1, 1)).to(device)
        mounts = torch.exp(
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(dim=1, keepdim=True)
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


class MkFillMaps(MkMapsBase):
    def __init__(self, sigma: float = 0.5) -> None:
        self.sigma = sigma

    def _mkmaps(
        self, boxes: YoloBoxes, hw: Tuple[int, int], original_hw: Tuple[int, int]
    ) -> Heatmaps:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return Heatmaps(heatmap)

        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64), torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        cxcy = boxes[:, :2] * img_wh
        box_wh = boxes[:, 2:]
        cx = cxcy[:, 0]
        cy = cxcy[:, 1]
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        grid_wh = img_wh.view(1, 2, 1, 1).expand_as(grid_xy)
        min_wh = (1.0 / img_wh.float()).view(1, 2).expand_as(box_wh)
        clamped_wh = torch.max(box_wh * 0.5 * self.sigma, min_wh)
        mounts = (
            (grid_xy.float() / grid_wh) - (grid_cxcy.float() / grid_wh)
        ).abs() < clamped_wh.view(box_count, 2, 1, 1).expand_as(grid_xy)
        mounts, _ = mounts.min(dim=1, keepdim=True)
        mounts = mounts.float()
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return Heatmaps(heatmap)


class Criterion:
    def __init__(
        self,
        box_weight: float = 4.0,
        heatmap_weight: float = 1.0,
        mkmaps: MkMapsFn = MkGaussianMaps(),
    ) -> None:
        self.box_weight = box_weight
        self.heatmap_weight = heatmap_weight

        self.hm_loss = HMLoss()
        self.box_loss = BoxLoss()
        self.mkmaps = mkmaps

    def __call__(
        self, images: ImageBatch, net_output: NetOutput, gt_boxes_list: List[YoloBoxes],
    ) -> Tuple[Tensor, Tensor, Tensor, Heatmaps]:
        anchormap, diffmaps, s_hms = net_output
        device = diffmaps.device
        box_losses: List[Tensor] = []
        hm_losses: List[Tensor] = []
        _, _, orig_h, orig_w = images.shape
        _, _, h, w = s_hms.shape

        t_hms = self.mkmaps(gt_boxes_list, (h, w), (orig_h, orig_w))
        hm_loss = self.hm_loss(s_hms, t_hms) * self.heatmap_weight

        for diffmap, heatmap, gt_boxes in zip(diffmaps, s_hms, gt_boxes_list):
            #  if len(gt_boxes) == 0:
            #      continue
            box_losses.append(
                self.box_loss(
                    anchormap=anchormap,
                    gt_boxes=gt_boxes,
                    diffmap=BoxMap(diffmap),
                    heatmap=Heatmap(heatmap),
                )
            )
        box_loss = torch.stack(box_losses).mean() * self.box_weight
        loss = hm_loss + box_loss
        return loss, hm_loss, box_loss, Heatmaps(t_hms)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
        visualize: Visualize,
        optimizer: Any,
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
                "train_hm",
                "test_loss",
                "test_box",
                "test_hm",
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
            self.model_loader.save_if_needed(
                self.model, self.meters[self.model_loader.key].get_value()
            )
            self.log()
            self.reset_meters()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for images, box_batch, ids, _ in tqdm(loader):
            images, box_batch = self.preprocess((images, box_batch))
            outputs = self.model(images)
            loss, hm_loss, box_loss, gt_hms = self.criterion(images, outputs, box_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.meters["train_loss"].update(loss.item())
            self.meters["train_box"].update(box_loss.item())
            self.meters["train_hm"].update(hm_loss.item())

    @torch.no_grad()
    def eval_one_epoch(self) -> None:
        self.model.eval()
        loader = self.test_loader
        for images, box_batch, ids, _ in tqdm(loader):
            images, box_batch = self.preprocess((images, box_batch))
            outputs = self.model(images)
            loss, hm_loss, box_loss, gt_hms = self.criterion(images, outputs, box_batch)
            self.meters["test_loss"].update(loss.item())
            self.meters["test_box"].update(box_loss.item())
            self.meters["test_hm"].update(hm_loss.item())
            preds = self.to_boxes(outputs)
            for (pred, gt) in zip(preds[0], box_batch):
                self.meters["score"].update(self.get_score(pred, gt))

        self.visualize(outputs, preds, box_batch, images, gt_hms)


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
        self.hflip_tta = HFlipTTA(to_boxes)
        self.vflip_tta = VFlipTTA(to_boxes)

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
