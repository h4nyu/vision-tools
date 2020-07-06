import typing as t
from torch import nn, Tensor
import torch
from scipy.optimize import linear_sum_assignment
from .utils import generalized_box_iou, box_cxcywh_to_xyxy
from typing_extensions import TypedDict
from typing import Tuple, List
from object_detection.entities import YoloBoxes, Labels


Pair = Tuple[YoloBoxes, Labels]
Target = TypedDict("Target", {"labels": Tensor, "boxes": Tensor})
Targets = t.List[Target]
MatchIndecies = t.List[t.Tuple[Tensor, Tensor]]


class HungarianMatcher:
    def __init__(
        self, cost_class: float = 1.0, cost_box: float = 1.0, cost_giou: float = 1.0,
    ) -> None:
        self.cost_class = cost_class
        self.cost_box = cost_box
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, preds: Pair, targets: Pair) -> MatchIndecies:
        ...
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
