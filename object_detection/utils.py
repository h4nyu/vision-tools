import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import random
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional
from torch import nn
from pathlib import Path
from torch import Tensor
from logging import getLogger
from .entities.box import (
    CoCoBoxes,
    YoloBoxes,
    PascalBoxes,
    Labels,
    yolo_to_coco,
    pascal_to_coco,
)

logger = getLogger(__name__)


def init_seed(seed: int = 777) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore


class DetectionPlot:
    def __init__(
        self,
        w: int = 128,
        h: int = 128,
        figsize: t.Tuple[int, int] = (10, 10),
        use_alpha: bool = True,
        show_probs: bool = False,
    ) -> None:
        self.w, self.h = (w, h)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.imshow(
            torch.ones(self.w, self.h, 3),
            interpolation="nearest",
        )
        self.use_alpha = use_alpha
        self.show_probs = show_probs

    def __del__(self) -> None:
        plt.close(self.fig)

    def save(self, path: t.Union[str, Path]) -> None:
        self.fig.savefig(path)

    def set_title(self, text: str) -> None:
        self.ax.set_title(text)

    def with_image(
        self, image: Tensor, alpha: Optional[float] = None
    ) -> None:
        if len(image.shape) == 2:
            h, w = image.shape
            if (h != self.h) and (w != self.w):
                image = (
                    F.interpolate(
                        image.unsqueeze(0).unsqueeze(0),
                        size=(self.h, self.w),
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            self.ax.imshow(
                image.detach().cpu(),
                interpolation="nearest",
                alpha=alpha,
            )
        elif len(image.shape) == 3:
            (
                _,
                h,
                w,
            ) = image.shape
            if (h != self.h) and (w != self.w):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.h, self.w),
                ).squeeze(0)
            image = image.permute(1, 2, 0)
            self.ax.imshow(
                image.detach().cpu(),
                interpolation="nearest",
                alpha=alpha,
            )
        else:
            shape = image.shape
            raise ValueError(f"invald shape={shape}")

    def with_yolo_boxes(
        self,
        boxes: YoloBoxes,
        probs: t.Optional[Tensor] = None,
        labels: t.Optional[Labels] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        self.with_coco_boxes(
            boxes=yolo_to_coco(boxes, size=(self.w, self.h)),
            probs=probs,
            color=color,
            labels=labels,
            fontsize=fontsize,
        )

    def with_pascal_boxes(
        self,
        boxes: PascalBoxes,
        probs: t.Optional[Tensor] = None,
        labels: t.Optional[Labels] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        self.with_coco_boxes(
            boxes=pascal_to_coco(boxes),
            probs=probs,
            color=color,
            labels=labels,
            fontsize=fontsize,
        )

    def with_coco_boxes(
        self,
        boxes: CoCoBoxes,
        probs: t.Optional[Tensor] = None,
        labels: t.Optional[Tensor] = None,
        color: str = "black",
        fontsize: int = 8,
    ) -> None:
        """
        boxes: coco format
        """
        b = len(boxes)
        _probs = probs if probs is not None else torch.ones((b,))
        _labels = labels if labels is not None else torch.zeros((b,), dtype=torch.int32)
        _boxes = boxes.clone()
        for box, p, c in zip(_boxes, _probs, _labels):
            x0 = box[0]
            y0 = box[1]
            self.ax.text(
                x0,
                y0,
                f"[{c}]{p:.2f}",
                fontsize=fontsize,
                color=color,
            )
            rect = mpatches.Rectangle(
                (x0, y0),
                width=box[2],
                height=box[3],
                fill=False,
                edgecolor=color,
                linewidth=1,
                alpha=float(p) if self.use_alpha else None,
            )
            self.ax.add_patch(rect)
