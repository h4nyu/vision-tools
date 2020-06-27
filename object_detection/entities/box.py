import torch
from typing import NewType, Tuple
from torch import Tensor
from .image import ImageSize

CoCoBoxes = NewType(
    "CoCoBoxes", Tensor
)  # [B, Pos] Pos:[x0, y0, width, height] original torch.int32
YoloBoxes = NewType(
    "YoloBoxes", Tensor
)  # [B, Pos] Pos:[cx, cy, width, height] normalized
PascalBoxes = NewType(
    "PascalBoxes", Tensor
)  # [B, Pos] Pos:[x0, y0, x1, y1] original torch.int32

Labels = NewType("Labels", Tensor)
Confidences = NewType("Confidences", Tensor)

PredBoxes = Tuple[CoCoBoxes, Confidences]
LabelBoxes = Tuple[CoCoBoxes, Labels]


def coco_to_yolo(coco: CoCoBoxes, size: ImageSize) -> YoloBoxes:
    size_w, size_h = size
    x0, y0, x1, y1 = coco_to_pascal(coco).unbind(-1)
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))


def coco_to_pascal(coco: CoCoBoxes) -> PascalBoxes:
    x0, y0, w, h = coco.unbind(-1)
    b = [x0, y0, x0 + w, y0 + h]
    return PascalBoxes(torch.stack(b, dim=-1))


def yolo_to_pascal(yolo: YoloBoxes, size: ImageSize) -> PascalBoxes:
    cx, cy, w, h = yolo.unbind(-1)
    size_w, size_h = size
    b = [
        (cx - 0.5 * w) * size_w,
        (cy - 0.5 * h) * size_h,
        (cx + 0.5 * w) * size_w,
        (cy + 0.5 * h) * size_h,
    ]
    return PascalBoxes(torch.stack(b, dim=-1))


def yolo_to_coco(yolo: YoloBoxes, size: ImageSize) -> CoCoBoxes:
    x0, y0, x1, y1 = yolo_to_pascal(yolo, size).unbind(-1)
    b = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1).long()
    return CoCoBoxes(b)


def pascal_to_yolo(pascal: PascalBoxes, size: ImageSize) -> YoloBoxes:
    x0, y0, x1, y1 = pascal.float().unbind(-1)
    size_w, size_h = size
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))


def pascal_to_coco(pascal: PascalBoxes) -> CoCoBoxes:
    x0, y0, x1, y1 = pascal.unbind(-1)
    b = [
        x0,
        y0,
        (x1 - x0),
        (y1 - y0),
    ]
    return CoCoBoxes(torch.stack(b, dim=-1))


def box_iou(a: YoloBoxes, b: YoloBoxes) -> Tensor:
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    iou = intersection / ua

    return iou
