import torch
from typing import *
from torch import Tensor
from PIL import Image, ImageDraw
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
from torchvision.utils import save_image

logger = getLogger(__name__)


def init_seed(seed: int = 777) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore


colors = [
    "green",
    "red",
    "blue",
]


class DetectionPlot:
    def __init__(
        self,
        img: torch.Tensor,
    ) -> None:
        if img.ndim == 2:
            ndarr = img.to('cpu', torch.uint8).numpy()
        if img.ndim == 3:
            ndarr = img.to('cpu', torch.uint8).permute(1, 2, 0).numpy()
        self.img = Image.fromarray(ndarr)
        self.draw = ImageDraw.Draw(self.img)

    def save(self, path: Union[str, Path]) -> None:
        self.img.save(path)

    def overlay(self, img: Tensor, alpha: float) -> None:
        other = Image.fromarray(img.to('cpu', torch.uint8).permute(1, 2, 0).numpy())
        self.img = Image.blend(self.img, other, alpha)

    @torch.no_grad()
    def draw_boxes(
        self,
        boxes: PascalBoxes,
        confidences: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        color: str = "white",
        font_size: int = 8,
        line_width: int = 1,
    ) -> None:
        _labels = labels.tolist() if labels is not None else []
        for i, box in enumerate(boxes.tolist()):
            self.draw.rectangle(box, width=line_width, outline=color)
            label = "{}".format(labels[i]) if labels is not None else ""
            confidence = (
                "{:.3f}".format(float(confidences[i]))
                if confidences is not None
                else ""
            )
            self.draw.text((box[0], box[1]), f"{label}: {confidence}", fill=color)
