import typing as t
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .matcher import HungarianMatcher, Outputs, Targets, MatchIndecies
from .utils import box_cxcywh_to_xyxy, generalized_box_iou
from app import config


Losses = t.TypedDict("Losses", {"box": Tensor, "label": Tensor,})


class SetCriterion(nn.Module):
    def __init__(
        self, num_classes: int, weights: t.Dict = {}, eos_coef: float = config.eos_coef
    ) -> None:
        super().__init__()
        self.matcher = HungarianMatcher()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: Outputs, targets: Targets) -> Tensor:
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        src_logits = outputs["pred_logits"]
        tgt_lables = [t["labels"] for t in targets]

        loss_label = self.loss_labels(src_logits, tgt_lables, indices)
        loss_box, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)
        #  loss_cardinality = self.loss_cardinality(outputs, targets,indices, num_boxes)
        return (
            loss_label * config.loss_label
            + config.loss_box * loss_box
            #  + loss_cardinality
            #  + config.loss_giou * loss_giou
        )

    def loss_cardinality(
        self, outputs: Outputs, targets: Targets, indices: MatchIndecies, num_boxes: int
    ) -> Tensor:
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err

    def loss_boxes(
        self, outputs: Outputs, targets: Targets, indices: MatchIndecies, num_boxes: int
    ) -> t.Tuple[Tensor, Tensor]:
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_box = (
            F.l1_loss(pred_boxes, target_boxes, reduction="none").sum() / num_boxes
        )
        loss_giou = (
            1
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        ).mean()
        return loss_box, loss_giou

    def loss_labels(
        self, src_logits: Tensor, tgt_lables: t.List[Tensor], indices: MatchIndecies
    ) -> Tensor:
        idx = self._get_src_permutation_idx(indices)
        tgt_classes_o = torch.cat([t[i] for t, (_, i) in zip(tgt_lables, indices)])

        # fill no-object class
        # the no-object class is the last class equal to self.num_classes
        tgt_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,  # type: ignore
        )

        # tgt_classes contains object class and no-object class
        tgt_classes[idx] = tgt_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), tgt_classes, self.empty_weight  # type: ignore
        )
        return loss_ce

    def _get_src_permutation_idx(
        self, indices: MatchIndecies
    ) -> t.Tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
