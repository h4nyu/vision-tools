import typing as t
import torch
from torch import Tensor


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    if x.shape[0] == 0:
        return x
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    if x.shape[0] == 0:
        return x
    x_c, y_c, w, h = x.unbind(-1)
    out = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(out, dim=-1)


BoxFmt = t.Literal["xyxy", "cxcywh"]


class Boxes:
    id: str
    w: float
    h: float
    boxes: Tensor
    confidences: Tensor
    fmt: BoxFmt

    def __init__(
        self,
        w: float,
        h: float,
        fmt: BoxFmt,
        boxes: Tensor,
        confidences: t.Optional[Tensor] = None,
        id: str = "",
    ) -> None:
        self.id = id
        self.w = w
        self.h = h
        self.boxes = boxes
        self.fmt = fmt
        self.confidences = (
            confidences if confidences is not None else torch.ones(boxes.shape[0])
        )

    def __len__(self,) -> int:
        return len(self.boxes)

    @property
    def device(self) -> t.Any:
        return self.boxes.device

    def to(self, device: t.Any) -> "Boxes":
        self.boxes = self.boxes.to(device)
        self.confidences = self.confidences.to(device)
        return self

    def to_cxcywh(self) -> "Boxes":
        if self.fmt == "xyxy":
            self.boxes = box_xyxy_to_cxcywh(self.boxes)
        return self

    def to_xyxy(self) -> "Boxes":
        if self.fmt == "cxcywh":
            self.boxes = box_cxcywh_to_xyxy(self.boxes)
        return self
