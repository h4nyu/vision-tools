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
BoxMaps = NewType("BoxMaps", Tensor)  # [B, 4, H, W]
BoxMap = NewType("BoxMap", Tensor)  # [4, H, W]


Labels = NewType("Labels", Tensor)
Confidences = NewType("Confidences", Tensor)

PredBoxes = Tuple[CoCoBoxes, Confidences]
LabelBoxes = Tuple[CoCoBoxes, Labels]

YoloBoxBatch = NewType("YoloBoxBatch", Tensor)  # [B, N, 4]
ConfidenceBatch = NewType("ConfidenceBatch", Tensor)  # [B, N] 0.0 ~ 1.0


def boxmap_to_boxes(am: BoxMap) -> YoloBoxes:
    return YoloBoxes(am.permute(2, 1, 0).reshape(-1, 4))


def coco_to_yolo(coco: CoCoBoxes, size: ImageSize) -> YoloBoxes:
    if len(coco) == 0:
        return YoloBoxes(coco)
    size_w, size_h = size
    x0, y0, x1, y1 = coco_to_pascal(coco).float().unbind(-1)
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))


def coco_to_pascal(coco: CoCoBoxes) -> PascalBoxes:
    if len(coco) == 0:
        return PascalBoxes(coco)
    x0, y0, w, h = coco.unbind(-1)
    b = [x0, y0, x0 + w, y0 + h]
    return PascalBoxes(torch.stack(b, dim=-1))


def yolo_to_pascal(yolo: YoloBoxes, wh: ImageSize) -> PascalBoxes:
    if len(yolo) == 0:
        return PascalBoxes(yolo)
    image_w, image_h = wh
    cx, cy, w, h = yolo.unbind(-1)
    size_w, size_h = wh
    b = [
        ((cx - w / 2) * size_w).clamp(min=0, max=image_w),
        ((cy - 0.5 * h) * size_h).clamp(min=0, max=image_h),
        ((cx + 0.5 * w) * size_w).clamp(min=0, max=image_w),
        ((cy + 0.5 * h) * size_h).clamp(min=0, max=image_h),
    ]
    return PascalBoxes(torch.stack(b, dim=-1).clamp(min=0.0))


def yolo_to_coco(yolo: YoloBoxes, size: ImageSize) -> CoCoBoxes:
    if len(yolo) == 0:
        return CoCoBoxes(yolo)
    x0, y0, x1, y1 = yolo_to_pascal(yolo, size).unbind(-1)
    b = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1).long()
    return CoCoBoxes(b)


def pascal_to_yolo(pascal: PascalBoxes, size: ImageSize) -> YoloBoxes:
    if len(pascal) == 0:
        return YoloBoxes(pascal)
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
    if len(pascal) == 0:
        return CoCoBoxes(pascal)
    x0, y0, x1, y1 = pascal.unbind(-1)
    b = [
        x0,
        y0,
        (x1 - x0),
        (y1 - y0),
    ]
    return CoCoBoxes(torch.stack(b, dim=-1))


def yolo_hflip(yolo: YoloBoxes) -> YoloBoxes:
    cx, cy, w, h = yolo.unbind(-1)
    b = [
        1.0 - cx,
        cy,
        w,
        h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))


def yolo_vflip(yolo: YoloBoxes) -> YoloBoxes:
    cx, cy, w, h = yolo.unbind(-1)
    b = [
        cx,
        1.0 - cy,
        w,
        h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))
