import torch
from torch import Tensor
from typing import Tuple, List, Dict, Any
import torch.nn.functional as F
from torchvision.ops import box_iou
import numpy as np


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
        thresholds: List[float] = [
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

    def update(self, pred_masks: Tensor, gt_masks: Tensor) -> float:
        value = self(pred_masks, gt_masks)
        self.num_samples += 1
        self.runing_value += value
        return value

    def update_batch(
        self, pred_masks: List[Tensor], gt_masks: List[Tensor]
    ) -> None:
        for p, g in zip(pred_masks, gt_masks):
            self.update(p, g)

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
        log: Dict[float, float] = {}
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
        thresholds: List[float] = [
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
    def value(self) -> Tuple[float, Dict[str, float]]:
        res = {}
        for k in self.running.keys():
            res[str(k)] = (
                self.running.get(k, 0) / self.num_samples
                if self.num_samples != 0
                else 0.0
            )
        mean = sum(res.values()) / len(res.values())
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

    def update(
        self, pred_box_batch: List[Tensor], gt_box_batch: List[Tensor]
    ) -> None:
        for p, g in zip(pred_box_batch, gt_box_batch):
            res = self(p, g)
            self.num_samples += 1
            for k in self.running.keys():
                self.running[k] = self.running.get(k, 0) + res.get(k, 0)

    @torch.no_grad()
    def __call__(self, pred_boxes: Tensor, gt_boxes: Tensor) -> Dict[float, float]:
        default = {k: 0.0 for k in self.running.keys()}
        if (len(gt_boxes) == 0) and (len(pred_boxes) > 0):
            return default

        if (len(pred_boxes) == 0) and (len(gt_boxes) > 0):
            return default

        if (len(pred_boxes) == 0) and (len(gt_boxes) == 0):
            return default

        res: Dict[float, float] = {}
        for th in self.running.keys():
            p = self.precision_at(
                pred_boxes=pred_boxes, gt_boxes=gt_boxes, threshold=th
            )
            res[th] = p
        return res


class BoxAP:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        num_classes: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.tp_list: List[Tensor] = []
        self.fp_list: List[Tensor] = []
        self.conf_list: List[Tensor] = []
        self.num_gts = 0

    def update(
        self,
        pred_box_batch: List[Tensor],
        pred_conf_batch: List[Tensor],
        gt_box_batch: List[Tensor],
    ) -> None:
        for pred_boxes, pred_confs, gt_boxes in zip(
            pred_box_batch, pred_conf_batch, gt_box_batch
        ):
            tps, fps = self.match_boxes(pred_boxes, gt_boxes, self.iou_threshold)
            self.tp_list.append(tps)
            self.fp_list.append(fps)
            self.conf_list.append(pred_confs)
            self.num_gts += gt_boxes.size(0)

    @staticmethod
    def match_boxes(
        pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float
    ) -> Tuple[Tensor, Tensor]:
        """
        test each predicted box (tp or fp).
        note: assume that pred_boxes are sorted by confidence descending.
        """
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        tps = torch.zeros(num_preds, dtype=torch.int)
        fps = torch.zeros(num_preds, dtype=torch.int)
        if num_preds > 0:
            if num_gts > 0:
                matched_gt = torch.zeros(num_gts, dtype=torch.bool)
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                max_ious = iou_matrix.max(axis=1).values
                max_idxs = iou_matrix.argmax(axis=1)
                for pred_idx, (max_iou, max_idx) in enumerate(zip(max_ious, max_idxs)):
                    if (max_iou >= iou_threshold) and ~matched_gt[max_idx]:
                        tps[pred_idx] = 1
                        matched_gt[max_idx] = True
                    else:
                        fps[pred_idx] = 1
            else:
                fps[:] = 1
        return tps, fps

    @property
    def value(self) -> Tuple[float, Dict[str, float]]:
        tps = torch.cat(self.tp_list, dim=0)
        fps = torch.cat(self.fp_list, dim=0)
        confs = torch.cat(self.conf_list, dim=0)
        precision, recall = precision_recall_curve(tps, fps, confs, self.num_gts)
        precision = torch.cummax(precision.flip(0), 0).values.flip(0)
        zero = torch.zeros(1, dtype=recall.dtype)
        diff_recall = torch.diff(recall, prepend=zero)
        return (precision * diff_recall).sum().item(), {}


def precision_recall_curve(
    tps: Tensor, fps: Tensor, confs: Tensor, num_gts: int
) -> Tuple[Tensor, Tensor]:
    confs, sort_idx = torch.sort(confs, descending=True)
    tps = tps[sort_idx]
    fps = fps[sort_idx]
    sum_tps = tps.cumsum(dim=0).float()
    sum_fps = fps.cumsum(dim=0).float()
    one = torch.scalar_tensor(1, dtype=torch.torch.float, device=tps.device)
    precision = sum_tps / torch.maximum(sum_tps + sum_fps, one)
    recall = (
        sum_tps / num_gts
        if num_gts > 0
        else torch.full_like(sum_tps, float("nan"), dtype=torch.float)
    )
    return precision, recall


class MeanBoxAP:
    def __init__(
        self,
        iou_thresholds: List[float] = [0.5],
        num_classes: int = 1,
    ):
        self.iou_thresholds = iou_thresholds
        self.num_classes = num_classes
        self.aps = [
            BoxAP(iou_threshold=iou_threshold, num_classes=num_classes)
            for iou_threshold in iou_thresholds
        ]
        self.reset()

    def reset(self) -> None:
        for ap in self.aps:
            ap.reset()

    def update(
        self,
        pred_box_batch: List[Tensor],
        pred_conf_batch: List[Tensor],
        gt_box_batch: List[Tensor],
    ) -> None:
        for ap in self.aps:
            ap.update(
                pred_box_batch=pred_box_batch,
                pred_conf_batch=pred_conf_batch,
                gt_box_batch=gt_box_batch,
            )

    @property
    def value(self) -> Tuple[float, Dict[str, float]]:
        aps = [ap.value for ap in self.aps]
        logs = {f"ap@{iou:.2f}": ap[0] for iou, ap in zip(self.iou_thresholds, aps)}
        ap = sum(ap[0] for ap in aps) / len(aps)
        return ap, logs


class MeanBoxF2:
    def __init__(
        self, iou_thresholds: List[float] = list(np.arange(0.3, 0.85, 0.05))
    ) -> None:
        self.beta = 2.0
        self.iou_thresholds = iou_thresholds
        self.corrects = [
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
        ] * len(iou_thresholds)

    def precision(self, tp: int, fp: int, fn: int) -> float:
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, tp: int, fp: int, fn: int) -> float:
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f_beat(self, precision: float, recall: float, beta: float) -> float:
        if precision + recall == 0:
            return 0.0
        return (
            (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)
        )

    @property
    def value(self) -> Tuple[float, Dict[str, float]]:
        precisions = [self.precision(**c) for c in self.corrects]
        recalls = [self.recall(**c) for c in self.corrects]
        f2s = [self.f_beat(p, r, self.beta) for p, r in zip(precisions, recalls)]
        f2 = np.mean(f2s)
        return f2, dict(
            f2=f2,
            precision=np.mean(precisions),
            recall=np.mean(recalls),
        )

    def update(
        self, pred_box_batch: List[Tensor], gt_box_batch: List[Tensor]
    ) -> None:
        for pred_boxes, gt_boxes in zip(pred_box_batch, gt_box_batch):
            for i, iou_threshold in enumerate(self.iou_thresholds):
                correct = self.correct_at_iou_thr(pred_boxes, gt_boxes, iou_threshold)
                self.corrects[i]["tp"] += correct["tp"]
                self.corrects[i]["fp"] += correct["fp"]
                self.corrects[i]["fn"] += correct["fn"]

    def reset(self) -> None:
        self.correct = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    @torch.no_grad()
    def correct_at_iou_thr(
        self, pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float
    ) -> Dict[str, int]:
        correct = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return correct

        elif len(gt_boxes) == 0:
            correct["fp"] = len(pred_boxes)
            return correct

        elif len(pred_boxes) == 0:
            correct["fn"] = len(gt_boxes)
            return correct

        iou_matrix = box_iou(pred_boxes, gt_boxes)
        num_preds, num_gt = iou_matrix.shape
        fp = torch.ones(num_gt, dtype=torch.bool)
        for ious in iou_matrix:
            iou, gt_idx = ious.max(dim=0)
            if iou >= iou_threshold:
                fp[gt_idx] = False
        fp_count = int(fp.sum())
        tp_count = num_gt - fp_count
        fn_count = num_preds - tp_count
        correct["tp"] = tp_count
        correct["fp"] = fp_count
        correct["fn"] = fn_count
        return correct
