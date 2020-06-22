import typing as t
from torch import nn, Tensor
import torch
from scipy.optimize import linear_sum_assignment
from .utils import generalized_box_iou, box_cxcywh_to_xyxy
from app import config
from typing_extensions import TypedDict

Outputs = TypedDict("Outputs", {"pred_logits": Tensor, "pred_boxes": Tensor})
Target = TypedDict("Target", {"labels": Tensor, "boxes": Tensor})
Targets = t.List[Target]
MatchIndecies = t.List[t.Tuple[Tensor, Tensor]]


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = config.cost_class,
        cost_box: float = config.cost_box,
        cost_giou: float = config.cost_giou,
    ) -> None:
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_box: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_box = cost_box
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Outputs, targets: Targets) -> MatchIndecies:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        batch_size, num_queries = pred_logits.shape[:2]

        out_probs = pred_logits.flatten(0, 1).softmax(-1)
        out_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets]).long()
        tgt_boxes = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_probs[:, tgt_ids]
        cost_box = torch.cdist(out_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes),
        )
        cost = (
            self.cost_box * cost_box
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost = cost.view(batch_size, num_queries, -1).cpu()
        tgt_sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(tgt_sizes, -1))]  # type: ignore
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
