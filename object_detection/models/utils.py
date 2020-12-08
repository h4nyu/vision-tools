import torch
import numpy as np
import typing as t
from functools import partial
from torch import nn, Tensor
from torchvision.ops.boxes import box_area


def round_filters(filters: t.Any, global_params: t.Any) -> t.Any:
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth,
        int(filters + divisor / 2) // divisor * divisor,
    )
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> t.Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def bias_init_with_prob(prior_prob: t.Any) -> t.Any:
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    if x.shape[0] == 0:
        return x
    x_c, y_c, w, h = x.unbind(-1)
    out = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h),
    ]
    return torch.stack(out, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    if x.shape[0] == 0:
        return x
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x: Tensor) -> Tensor:
    x0, y0, w, h = x.unbind(-1)
    b = [x0, y0, (x0 + w), (y0 + h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    #  assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    #  assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class NestedTensor:
    def __init__(
        self,
        tensors: Tensor,
        mask: t.Optional[Tensor] = None,
    ) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: t.Any) -> "NestedTensor":
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        cast_mask: t.Optional[Tensor] = None
        if mask is not None:
            cast_mask = mask.to(device)
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(
        self,
    ) -> t.Tuple[Tensor, t.Optional[Tensor]]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)
