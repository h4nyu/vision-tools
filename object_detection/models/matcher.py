from torch import nn, Tensor
import torch
from scipy.optimize import linear_sum_assignment
from .utils import generalized_box_iou, box_cxcywh_to_xyxy
from typing_extensions import TypedDict
from object_detection import (
    YoloBoxBatch,
    ConfidenceBatch,
    YoloBoxes,
    Labels,
)


Preds = tuple[YoloBoxBatch, ConfidenceBatch]
Targets = list[tuple[list[YoloBoxes], list[Labels]]]
MatchIndecies = list[tuple[Tensor, Tensor]]


class NearnestMatcher:
    def __call__(
        self,
        pred: YoloBoxes,
        gt: YoloBoxes,
        size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        w, h = size
        eps_dist = ((1 / w) ** 2 + (1 / w) ** 2) ** (1 / 2)
        pred_count = pred.shape[0]
        gt_count = gt.shape[0]
        pred_ctr = (
            pred[:, :2]
            .view(pred_count, 1, 2)
            .expand(
                pred_count,
                gt_count,
                2,
            )
        )
        gt_ctr = gt[:, :2]
        matrix = ((pred_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        min_dist, matched_idx = matrix.min(dim=1)
        max_lenght = (gt[:, 2:] / 2).min(dim=1)[0][matched_idx]
        filter_idx = min_dist < max_lenght.clamp(min=eps_dist)
        return matched_idx, filter_idx


class CenterMatcher:
    def __call__(
        self,
        pred: YoloBoxes,
        gt: YoloBoxes,
        size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        w, h = size
        pixcel_dist = ((1 / w) ** 2 + (1 / h) ** 2) ** (1 / 2) / 2
        pred_count = pred.shape[0]
        gt_count = gt.shape[0]
        pred_ctr = (
            pred[:, :2]
            .view(pred_count, 1, 2)
            .expand(
                pred_count,
                gt_count,
                2,
            )
        )
        gt_ctr = gt[:, :2]
        matrix = ((pred_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        min_dist, matched_idx = matrix.min(dim=1)
        max_lenght = (gt[:, 2:] / 2).min(dim=1)[0][matched_idx]
        filter_idx = min_dist < pixcel_dist
        return matched_idx, filter_idx


class HungarianMatcher:
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_box: float = 1.0,
        cost_giou: float = 1.0,
    ) -> None:
        self.cost_class = cost_class
        self.cost_box = cost_box
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, preds: Preds, targets: Targets) -> None:
        pred_box_batch, pred_cfd_batch = preds
        #  pred_boxes, pred_logits = preds

        #  batch_size, num_queries = pred_logits.shape[:2]
        #  out_probs = pred_logits.flatten(0, 1).softmax(-1)
        #  out_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        #  tgt_ids = torch.cat([v["labels"] for v in targets]).long()
        #  tgt_boxes = torch.cat([v["boxes"] for v in targets])
        #
        #  cost_class = -out_probs[:, tgt_ids]
        #  cost_box = torch.cdist(out_boxes, tgt_boxes, p=1)
        #  cost_giou = -generalized_box_iou(
        #      box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes),
        #  )
        #  cost = (
        #      self.cost_box * cost_box
        #      + self.cost_class * cost_class
        #      + self.cost_giou * cost_giou
        #  )
        #  cost = cost.view(batch_size, num_queries, -1).cpu()
        #  tgt_sizes = [len(v["boxes"]) for v in targets]
        #
        #  indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(tgt_sizes, -1))]  # type: ignore
        #  return [
        #      (
        #          torch.as_tensor(i, dtype=torch.int64),
        #          torch.as_tensor(j, dtype=torch.int64),
        #      )
        #      for i, j in indices
        #  ]
