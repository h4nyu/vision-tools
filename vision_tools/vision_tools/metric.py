import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import box_iou


class MaskIou:
    def __init__(self, use_batch: bool = False):
        self.use_batch = use_batch

    def _simple(self, pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
        iou_rows = []
        pred_masks = pred_masks.bool().contiguous().view(pred_masks.shape[0], -1)
        gt_masks = gt_masks.bool().contiguous().view(gt_masks.shape[0], -1)
        for pred_mask in pred_masks:
            intersection = (gt_masks & pred_mask).sum(dim=-1)
            union = (gt_masks | pred_mask).sum(dim=-1)
            iou_row = intersection / union
            iou_rows.append(iou_row)
        iou_matrix = torch.stack(iou_rows).nan_to_num(nan=0)
        return iou_matrix

    def _batch(self, pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
        pred_masks = pred_masks.bool().contiguous().view(pred_masks.shape[0], 1, -1)
        gt_masks = gt_masks.bool().contiguous().view(gt_masks.shape[0], -1)
        intersection = (pred_masks & gt_masks).sum(dim=-1)
        union = (pred_masks | gt_masks).sum(dim=-1)
        iou_matrix = intersection / union
        iou_matrix = iou_matrix.nan_to_num(nan=0)
        return iou_matrix

    def __call__(self, pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
        if self.use_batch:
            return self._batch(pred_masks, gt_masks)
        else:
            return self._simple(pred_masks, gt_masks)


class MaskAP:
    def __init__(
        self,
        reduce_size: int = 1,
        use_batch: bool = False,
        thresholds: list[float] = [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
    ):
        self.thresholds = thresholds
        self.reduce_size = reduce_size
        self.mask_iou = MaskIou(use_batch=use_batch)
        self.num_samples = 0
        self.runing_value = 0.0

    @property
    def value(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.runing_value / self.num_samples

    def precision_at(
        self, pred_masks: Tensor, gt_masks: Tensor, threshold: float
    ) -> float:
        iou_matrix = self.mask_iou(pred_masks, gt_masks)
        num_preds, num_gt = iou_matrix.shape
        fp = torch.ones(num_gt, dtype=torch.bool)
        for ious in iou_matrix:
            iou, gt_idx = ious.max(dim=0)
            if iou >= threshold:
                fp[gt_idx] = False
        fp_count = fp.sum()
        tp_count = num_gt - fp_count
        res = tp_count / (num_gt + num_preds - tp_count)
        return res.item()

    def accumulate(self, pred_masks: Tensor, gt_masks: Tensor) -> float:
        value = self(pred_masks, gt_masks)
        self.num_samples += 1
        self.runing_value += value
        return value

    def accumulate_batch(
        self, pred_masks: list[Tensor], gt_masks: list[Tensor]
    ) -> None:
        for p, g in zip(pred_masks, gt_masks):
            self.accumulate(p, g)

    @torch.no_grad()
    def __call__(self, pred_masks: Tensor, gt_masks: Tensor) -> float:
        if (len(gt_masks) == 0) and (len(pred_masks) > 0):
            return 0.0

        if (len(pred_masks) == 0) and (len(gt_masks) > 0):
            return 0.0

        if (len(pred_masks) == 0) and (len(gt_masks) == 0):
            return 1.0

        if self.reduce_size > 1:
            split_idx = pred_masks.shape[0]
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0).float(),
                scale_factor=1 / self.reduce_size,
            )[0].bool()
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(0).float(),
                scale_factor=1 / self.reduce_size,
            )[0].bool()
        running_p = 0.0
        log: dict[float, float] = {}
        for th in self.thresholds:
            p = self.precision_at(
                pred_masks=pred_masks, gt_masks=gt_masks, threshold=th
            )
            running_p += p
            log[th] = p
        return running_p / len(self.thresholds)


class BoxMAP:
    def __init__(
        self,
        thresholds: list[float] = [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
    ):
        self.thresholds = thresholds
        self.num_samples = 0
        self.running = {k: 0.0 for k in thresholds}

    def reset(self) -> None:
        self.running = {k: 0.0 for k in self.thresholds}
        self.num_samples = 0

    @property
    def value(self) -> tuple[float, dict[str, float]]:
        res = {}
        for k in self.running.keys():
            res[str(k)] = (
                self.running.get(k, 0) / self.num_samples
                if self.num_samples != 0
                else 0.0
            )
        mean = sum(self.running.values()) / len(self.running.keys())
        return mean, res

    def precision_at(
        self, pred_boxes: Tensor, gt_boxes: Tensor, threshold: float
    ) -> float:
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        num_preds, num_gt = iou_matrix.shape
        fp = torch.ones(num_gt, dtype=torch.bool)
        for ious in iou_matrix:
            iou, gt_idx = ious.max(dim=0)
            if iou >= threshold:
                fp[gt_idx] = False
        fp_count = fp.sum()
        tp_count = num_gt - fp_count
        res = tp_count / (num_gt + num_preds - tp_count)
        return res.item()

    def accumulate(
        self, pred_box_batch: list[Tensor], gt_box_batch: list[Tensor]
    ) -> None:
        for p, g in zip(pred_box_batch, gt_box_batch):
            res = self(p, g)
            self.num_samples += 1
            for k in self.running.keys():
                self.running[k] = self.running.get(k, 0) + res.get(k, 0)

    @torch.no_grad()
    def __call__(self, pred_boxes: Tensor, gt_boxes: Tensor) -> dict[float, float]:
        default = {k: 0.0 for k in self.running.keys()}
        if (len(gt_boxes) == 0) and (len(pred_boxes) > 0):
            return default

        if (len(pred_boxes) == 0) and (len(gt_boxes) > 0):
            return default

        if (len(pred_boxes) == 0) and (len(gt_boxes) == 0):
            return default

        res: dict[float, float] = {}
        for th in self.running.keys():
            p = self.precision_at(
                pred_boxes=pred_boxes, gt_boxes=gt_boxes, threshold=th
            )
            res[th] = p
        return res
