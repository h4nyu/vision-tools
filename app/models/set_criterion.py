import typing as t
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .matcher import HungarianMatcher, Outputs, Targets, MatchIndecies



class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, weights: t.Dict = {}) -> None:
        super().__init__()
        self.matcher = HungarianMatcher()
        self.num_classes = num_classes

    def forward(self, outputs: Outputs, targets: Targets) -> Tensor:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        loss_labels = self.loss_labels(outputs, targets, indices, num_boxes)
        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
        loss_cardinality = self.loss_cardinality(outputs, targets, indices, num_boxes)

        return loss_labels * 2 + loss_boxes * 1 + loss_cardinality * 1

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
    ) -> Tensor:
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss = F.l1_loss(pred_boxes, target_boxes, reduction="none").sum() / num_boxes
        return loss

    def loss_labels(
        self, outputs: Outputs, targets: Targets, indices: MatchIndecies, num_boxes: int
    ) -> Tensor:
        pred_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes - 1,  # ????
            dtype=torch.int64,
            device=pred_logits.device,  # type: ignore
        )
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
        return loss

    def _get_src_permutation_idx(
        self, indices: MatchIndecies
    ) -> t.Tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
