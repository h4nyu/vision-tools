import torch
import typing
from torch import Tensor


class ClosestAssign:
    """
    select k anchors whose center are closest to
    the center of ground-truth based on L2 distance.
    """

    def __init__(self, topk: int) -> None:
        self.topk = topk

    def __call__(self, anchor: Tensor, gt: Tensor) -> Tensor:
        device = anchor.device
        gt_count = gt.shape[0]
        anchor_count = anchor.shape[0]
        if gt_count == 0:
            return torch.zeros((0, gt_count), device=device)
        anchor_ctr = (
            ((anchor[:, :2] + anchor[:, 2:]) / 2.0)
            .view(anchor_count, 1, 2)
            .expand(
                anchor_count,
                gt_count,
                2,
            )
        )
        gt_ctr = gt[:, :2]
        matrix = ((anchor_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        _, matched_idx = torch.topk(matrix, self.topk, dim=0, largest=False)
        return matched_idx.t()
