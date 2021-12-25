from torch import nn, Tensor
import torch


Preds = tuple[Tensor, Tensor]
Targets = list[tuple[list[Tensor], list[Tensor]]]
MatchIndecies = list[tuple[Tensor, Tensor]]


class NearnestMatcher:
    def __call__(
        self,
        pred: Tensor,
        gt: Tensor,
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
        pred: Tensor,
        gt: Tensor,
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