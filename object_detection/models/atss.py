import torch
from torch import Tensor
from .closest_assign import ClosestAssign
from torchvision.ops.boxes import box_iou
from object_detection.entities import PascalBoxes, AnchorMap
import typing


class ATSS:
    """
    Adaptive Training Sample Selection
    """

    def __init__(
        self,
        topk: int = 9,
        pylamid_levels: typing.List[int] = [3, 4, 5],
    ) -> None:
        self.topk = topk
        self.pylamid_levels = pylamid_levels
        self.closest_assign = ClosestAssign(topk)

    def __call__(
        self,
        anchors: PascalBoxes,
        gt: PascalBoxes,
    ) -> Tensor:
        matched_ids = self.closest_assign(anchors, gt)
        gt_count, anchor_count = matched_ids.shape
        pos_ids_list: typing.List[Tensor] = []
        for i in range(gt_count):
            ids = matched_ids[i]
            matched_anchors = anchors[ids]
            ious = box_iou(matched_anchors, gt[[i]]).view(-1)
            m_iou = ious.mean()
            s_iou = ious.std()
            th = m_iou + s_iou
            pos_ids = ids[ious >= th]
            pos_ids_list.append(pos_ids)
        res = torch.unique(torch.stack(pos_ids_list))
        return res
