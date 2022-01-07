import torch
import typing
from torch import Tensor
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F


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


class SimOTA:
    def __init__(
        self,
        topk: int,
        radius: float = 2.5,
        box_weight: float = 3.0,
    ) -> None:
        self.topk = topk
        self.radius = radius
        self.box_weight = box_weight

    def candidates(
        self,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gt_boxes = gt_boxes.unsqueeze(1)
        gt_centers = (gt_boxes[:, :, 0:2] + gt_boxes[:, :, 2:4]) / 2.0

        is_in_box = (  # grid cell inside gt-box
            (gt_boxes[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] <= gt_boxes[:, :, 2])
            & (gt_boxes[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] <= gt_boxes[:, :, 3])
        )  # [num_gts, num_proposals]
        gt_center_lbound = gt_centers - self.radius * strides.unsqueeze(1)
        gt_center_ubound = gt_centers + self.radius * strides.unsqueeze(1)

        is_in_center = (  # grid cell near gt-box center
            (gt_center_lbound[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] <= gt_center_ubound[:, :, 0])
            & (gt_center_lbound[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] <= gt_center_ubound[:, :, 1])
        )  # [num_gts, num_proposals]
        candidates = (is_in_box | is_in_center).any(dim=0)
        center_matrix = (is_in_box & is_in_center)[
            :, candidates
        ]  # [num_gts, num_fg_candidates]
        return candidates, center_matrix

    @torch.no_grad()
    def __call__(
        self,
        anchor_points: Tensor,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        gt_boxes: Tensor,
        strides: Tensor,
    ) -> Tensor:  # [gt_index, pred_index]
        device = pred_boxes.device
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        if gt_count == 0 or pred_count == 0:
            return torch.zeros(0, 2).to(device)
        candidates, center_matrix = self.candidates(
            anchor_points=anchor_points,
            gt_boxes=gt_boxes,
            strides=strides,
        )
        with torch.cuda.amp.autocast(enabled=False):
            score_matrix = -F.binary_cross_entropy(
                pred_scores[candidates],
                torch.ones(int(candidates.sum())).to(device),
                reduction="none",
            ).expand(gt_count, -1)
        iou_matrix = box_iou(gt_boxes, pred_boxes[candidates])
        matrix = score_matrix + self.box_weight * iou_matrix + center_matrix * 10000
        topk = min(self.topk, pred_count)
        topk_ious, _ = torch.topk(iou_matrix, topk, dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp(min=1)
        matching_matrix = torch.zeros((gt_count, pred_count), dtype=torch.long)
        candidate_idx = candidates.nonzero().view(-1)
        for (row, dynamic_topk, matching_row) in zip(
            matrix, dynamic_ks, matching_matrix
        ):
            _, pos_idx = torch.topk(row, k=dynamic_topk)
            matching_row[candidate_idx[pos_idx]] = 1
        return matching_matrix.nonzero()
