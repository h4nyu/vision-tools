import torch
import typing
from torch import Tensor
from object_detection.entities import YoloBoxes


class ClosestAssign:
    """
    select k anchors whose center are closest to
    the center of ground-truth based on L2 distance.
    """

    def __init__(self, topk: int) -> None:
        self.topk = topk

    def __call__(self, anchor: YoloBoxes, gt: YoloBoxes) -> Tensor:
        anchor_count = anchor.shape[0]
        gt_count = gt.shape[0]
        anchor_ctr = (
            anchor[:, :2]
            .view(anchor_count, 1, 2)
            .expand(
                anchor_count,
                gt_count,
                2,
            )
        )
        gt_ctr = gt[:, :2]
        matrix = ((anchor_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        _, matched_idx = torch.topk(
            matrix, self.topk, dim=0, largest=False
        )
        return matched_idx
