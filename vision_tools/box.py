import torch
from typing import *
from torch import Tensor
from torchvision.ops.boxes import box_iou, box_area

Number = Union[float, int]


def boxmap_to_boxes(x: Tensor) -> Tensor:
    return Tensor(x.permute(2, 1, 0).reshape(-1, 4))


def boxmaps_to_boxes(x: Tensor) -> Tensor:
    return Tensor(x.permute(3, 2, 0, 1).reshape(-1, 4))


def resize_boxes(boxes: Tensor, scale: tuple[float, float]) -> Tensor:
    if len(boxes) == 0:
        return boxes
    wr, hr = scale
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [
        x0 * wr,
        y0 * hr,
        x1 * wr,
        y1 * hr,
    ]
    return Tensor(torch.stack(b, dim=-1))


def coco_to_yolo(coco: Tensor, size: tuple[int, int]) -> Tensor:
    if len(coco) == 0:
        return Tensor(coco)
    size_w, size_h = size
    x0, y0, x1, y1 = coco_to_pascal(coco).float().unbind(-1)
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return Tensor(torch.stack(b, dim=-1))


def coco_to_pascal(coco: Tensor) -> Tensor:
    if len(coco) == 0:
        return Tensor(coco)
    x0, y0, w, h = coco.unbind(-1)
    b = [x0, y0, x0 + w, y0 + h]
    return Tensor(torch.stack(b, dim=-1))


def yolo_to_pascal(yolo: Tensor, wh: tuple[int, int]) -> Tensor:
    if len(yolo) == 0:
        return Tensor(yolo)
    image_w, image_h = wh
    cx, cy, w, h = yolo.unbind(-1)
    size_w, size_h = wh
    b = [
        ((cx - 0.5 * w) * size_w),
        ((cy - 0.5 * h) * size_h),
        ((cx + 0.5 * w) * size_w),
        ((cy + 0.5 * h) * size_h),
    ]
    return Tensor(torch.stack(b, dim=-1))


def yolo_to_coco(yolo: Tensor, size: tuple[int, int]) -> Tensor:
    if len(yolo) == 0:
        return yolo
    x0, y0, x1, y1 = yolo_to_pascal(yolo, size).unbind(-1)
    b = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)
    return b


def pascal_to_yolo(pascal: Tensor, size: tuple[int, int]) -> Tensor:
    if len(pascal) == 0:
        return Tensor(pascal)
    x0, y0, x1, y1 = pascal.float().unbind(-1)
    size_w, size_h = size
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return Tensor(torch.stack(b, dim=-1))


def pascal_to_coco(pascal: Tensor) -> Tensor:
    if len(pascal) == 0:
        return pascal
    x0, y0, x1, y1 = pascal.unbind(-1)
    b = [
        x0,
        y0,
        (x1 - x0),
        (y1 - y0),
    ]
    return torch.stack(b, dim=-1)


def yolo_hflip(yolo: Tensor) -> Tensor:
    if len(yolo) == 0:
        return yolo
    cx, cy, w, h = yolo.unbind(-1)
    b = [
        1.0 - cx,
        cy,
        w,
        h,
    ]
    return Tensor(torch.stack(b, dim=-1))


def yolo_vflip(yolo: Tensor) -> Tensor:
    cx, cy, w, h = yolo.unbind(-1)
    b = [
        cx,
        1.0 - cy,
        w,
        h,
    ]
    return Tensor(torch.stack(b, dim=-1))


def yolo_clamp(yolo: Tensor) -> Tensor:
    return pascal_to_yolo(
        Tensor(yolo_to_pascal(yolo, (1, 1)).clamp(min=0.0, max=1.0)),
        (1, 1),
    )


def box_clamp(boxes: Tensor, width: int, height: int) -> Tensor:
    if len(boxes) == 0:
        return boxes
    x0, y0, x1, y1 = boxes.clamp(min=0).unbind(-1)
    x0 = x0.clamp(max=width)
    x1 = x1.clamp(max=width)
    y0 = y0.clamp(max=height)
    y1 = y1.clamp(max=height)
    return Tensor(torch.stack([x0, y0, x1, y1], dim=-1))


def shift(boxes: Tensor, diff: tuple[Number, Number]) -> Tensor:
    if len(boxes) == 0:
        return boxes
    diff_x, diff_y = diff
    boxes[:, [0, 2]] = boxes[:, [0, 2]] + diff_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] + diff_y
    return Tensor(boxes)


def filter_size(
    boxes: Tensor, cond: Callable[[Tensor], Tensor]
) -> tuple[Tensor, Tensor]:
    if len(boxes) == 0:
        return boxes, torch.tensor([], dtype=torch.bool)
    x0, y0, x1, y1 = boxes.unbind(-1)
    area = (x1 - x0) * (y1 - y0)
    indices = cond(area)
    return Tensor(boxes[indices]), indices


def box_in_area(
    boxes: Tensor,
    area: Tensor,
    min_fill: float = 0.7,
) -> Tensor:
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    lt = torch.max(boxes[:, :2], area[:2])
    rb = torch.min(boxes[:, 2:], area[2:])
    wh = (rb - lt).clamp(min=0)
    overlaped_area = wh[:, 0] * wh[:, 1]
    areas = box_area(boxes)
    fill_ratio = overlaped_area / areas
    indices = fill_ratio > min_fill
    return indices


def box_hflip(boxes: Tensor, image_size: tuple[Number, Number]) -> Tensor:
    if len(boxes) == 0:
        return boxes
    w, h = image_size
    box_w = boxes[:, 2] - boxes[:, 0]
    boxes[:, 0] = w - boxes[:, 0] - box_w
    boxes[:, 2] = w - boxes[:, 2] + box_w
    return Tensor(boxes)


def box_vflip(boxes: Tensor, image_size: tuple[Number, Number]) -> Tensor:
    if len(boxes) == 0:
        return boxes
    w, h = image_size
    box_h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 1] = h - boxes[:, 1] - box_h
    boxes[:, 3] = h - boxes[:, 3] + box_h
    return Tensor(boxes)


def box_padding(boxes: Tensor, offset: Number) -> Tensor:
    if len(boxes) == 0:
        return Tensor(boxes)
    x0, y0, x1, y1 = boxes.unbind(-1)
    return Tensor(
        torch.stack(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            dim=-1,
        )
    )


def to_center_points(boxes: Tensor) -> Tensor:
    if len(boxes) == 0:
        return boxes

    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(
        [
            (x0 + x1) / 2,
            (y0 + y1) / 2,
        ],
        dim=-1,
    )


def filter_limit(
    boxes: Tensor,
    confidences: Tensor,
    labels: Tensor,
    limit: int,
) -> tuple[Tensor, Tensor, Tensor]:
    unique_labels = torch.unique(labels)
    box_list = []
    label_list = []
    conf_list = []
    for c in unique_labels:
        c_indecies = labels == c
        c_boxes = boxes[c_indecies][:limit]
        c_labels = labels[c_indecies][:limit]
        c_confidences = confidences[c_indecies][:limit]
        box_list.append(c_boxes)
        label_list.append(c_labels)
        conf_list.append(c_confidences)
    return (
        torch.cat(box_list) if len(box_list) > 0 else boxes,
        torch.cat(conf_list) if len(conf_list) > 0 else confidences,
        torch.cat(label_list) if len(label_list) > 0 else labels,
    )
