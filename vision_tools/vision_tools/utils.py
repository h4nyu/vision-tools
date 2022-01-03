from torchvision.utils import (
    draw_segmentation_masks,
    save_image,
    draw_bounding_boxes,
    make_grid,
)
from torchvision.ops import masks_to_boxes
import torch
from typing import *
from torch import Tensor
import json
import random
import numpy as np
import torch.nn.functional as F
from typing import Optional, Callable
from torch import nn
from pathlib import Path
from torch import Tensor
from logging import getLogger
from vision_tools import (
    Number,
    resize_points,
)
from torchvision.utils import save_image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf


logger = getLogger(__name__)


def seed_everything(seed: int = 777) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ToDevice:
    def __init__(
        self,
        device: str,
    ) -> None:
        self.device = device

    def __call__(
        self, *args: Union[Tensor, list[Tensor]], **kargs: Union[Tensor, list[Tensor]]
    ) -> Any:
        if kargs is not None:
            return {
                k: [i.to(self.device) for i in v]
                if isinstance(v, list)
                else v.to(self.device)
                for k, v in kargs.items()
            }
        return tuple(
            [i.to(self.device) for i in x] if isinstance(x, list) else x.to(self.device)
            for x in args
        )


colors = [
    "green",
    "red",
    "blue",
]

T = TypeVar("T", bound=nn.Module)


class Comparator(Protocol):
    def __call__(self, prev: float, next: float) -> bool:
        ...


class Checkpoint(Generic[T]):
    def __init__(
        self,
        root_path: str,
        default_score: float,
        comparator: Comparator = lambda p, n: p <= n,
    ) -> None:
        self.root_path = Path(root_path)
        self.model_path = self.root_path.joinpath("checkpoint.pth")
        self.checkpoint_path = self.root_path.joinpath("checkpoint.yaml")
        self.default_score = default_score
        self.root_path.mkdir(exist_ok=True)
        self.score = default_score
        self.comparator = comparator

    def load_if_exists(self, model: T) -> tuple[T, float]:
        if self.model_path.exists() and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.model_path))
            conf = OmegaConf.load(self.checkpoint_path)
            score = conf.get("score", self.default_score)  # type: ignore
            self.score = score
            return model, score
        else:
            return model, self.default_score

    def save_if_needed(self, model: T, score: float) -> None:
        if self.comparator(self.score, score):
            torch.save(model.state_dict(), self.model_path)  # type: ignore
            OmegaConf.save(config=dict(score=score), f=self.checkpoint_path)
            self.score == score


@torch.no_grad()
def draw(
    image: Tensor,
    masks: Optional[Tensor] = None,
    boxes: Optional[Tensor] = None,
) -> Tensor:
    image = image.detach().to("cpu").float()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if masks is not None and len(masks) > 0:
        empty_filter = masks.sum(dim=[1, 2]) > 0
        masks = masks[empty_filter]
        masks = masks.to("cpu")
        plot = draw_segmentation_masks((image * 255).to(torch.uint8), masks, alpha=0.3)
        boxes = masks_to_boxes(masks)
        return draw_bounding_boxes(plot, boxes)
    elif boxes is not None and len(boxes) > 0:
        return draw_bounding_boxes((image * 255).to(torch.uint8), boxes)
    else:
        return image


@torch.no_grad()
def batch_draw(
    image_batch: Tensor,
    box_batch: Optional[list[Tensor]] = None,
    label_batch: Optional[list[Tensor]] = None,
) -> Tensor:
    if box_batch is not None:
        return make_grid(
            [
                draw(image=image, boxes=boxes)
                for image, boxes in zip(image_batch, box_batch)
            ]
        )
    return make_grid(image_batch)

    # image = image.detach().to("cpu").float()
    # if image.shape[0] == 1:
    #     image = image.expand(3, -1, -1)
    # if masks is not None and len(masks) > 0:
    #     empty_filter = masks.sum(dim=[1, 2]) > 0
    #     masks = masks[empty_filter]
    #     masks = masks.to("cpu")
    #     plot = draw_segmentation_masks((image * 255).to(torch.uint8), masks, alpha=0.3)
    #     boxes = masks_to_boxes(masks)
    #     plot = draw_bounding_boxes(plot, boxes)
    #     plot = plot / 255
    # elif boxes is not None and len(boxes) > 0:
    #     plot = draw_bounding_boxes((image * 255).to(torch.uint8), boxes)
    #     plot = plot / 255
    # else:
    #     plot = image
    # save_image(plot, path)
