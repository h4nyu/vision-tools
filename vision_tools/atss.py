import torch
from torch import Tensor
from .assign import ClosestAssign
from torchvision.ops.boxes import box_iou
from vision_tools import Boxes, AnchorMap


class ATSS:
    """
    Adaptive Training Sample Selection
    """

    def __init__(
        self,
        topk: int = 9,
    ) -> None:
        self.topk = topk
        self.assign = ClosestAssign(topk)

    def __call__(
        self,
        anchors: Boxes,
        gt: Boxes,
    ) -> Tensor:
        device = anchors.device
        matched_ids = self.assign(anchors, gt)
        gt_count, _ = matched_ids.shape
        anchor_count, _ = anchors.shape
        pos_ids = torch.zeros(
            (
                gt_count,
                anchor_count,
            ),
            device=device,
        )
        for i in range(gt_count):
            ids = matched_ids[i]
            matched_anchors = anchors[ids]
            ious = box_iou(matched_anchors, gt[[i]]).view(-1)
            m_iou = ious.mean()
            s_iou = ious.std()
            th = m_iou + s_iou
            pos_ids[i, ids[ious > th]] = True
        return torch.nonzero(pos_ids, as_tuple=False)
