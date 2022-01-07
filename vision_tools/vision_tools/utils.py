from torchvision.utils import (
    draw_segmentation_masks,
    save_image,
    draw_bounding_boxes,
    make_grid,
)
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision.ops import masks_to_boxes, box_convert
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


# wait torchvision 0.12 release
@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.
    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return (
        torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
    )


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
        self.checkpoint_model_path = self.root_path.joinpath("checkpoint.pth")
        self.current_model_path = self.root_path.joinpath("current.pth")
        self.checkpoint_path = self.root_path.joinpath("checkpoint.yaml")
        self.default_score = default_score
        self.root_path.mkdir(exist_ok=True)
        self.score = default_score
        self.comparator = comparator

    def load_if_exists(self, model: T, target: str = "checkpoint") -> tuple[T, float]:
        model_path = (
            self.checkpoint_model_path
            if target == "checkpoint"
            else self.current_model_path
        )
        if model_path.exists() and model_path.exists():
            model.load_state_dict(torch.load(model_path))
            conf = OmegaConf.load(self.checkpoint_path)
            score = conf.get("score", self.default_score)  # type: ignore
            self.score = score
            return model, score
        else:
            return model, self.default_score

    def save_if_needed(self, model: T, score: float) -> None:
        if self.comparator(self.score, score):
            torch.save(model.state_dict(), self.checkpoint_model_path)  # type: ignore
            OmegaConf.save(config=dict(score=score), f=self.checkpoint_path)
            self.score == score
        torch.save(model.state_dict(), self.current_model_path)  # type: ignore


@torch.no_grad()
def draw(
    image: Tensor,
    masks: Optional[Tensor] = None,
    boxes: Optional[Tensor] = None,
    points: Optional[Tensor] = None,
) -> Tensor:
    image = image.detach().to("cpu").float()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if masks is None and boxes is None and points is None:
        return image
    plot = (image * 255).to(torch.uint8)
    if masks is not None and len(masks) > 0:
        empty_filter = masks.sum(dim=[1, 2]) > 0
        masks = masks[empty_filter]
        masks = masks.to("cpu")
        plot = draw_segmentation_masks(plot, masks, alpha=0.3)
        boxes = masks_to_boxes(masks)
        plot = draw_bounding_boxes(plot, boxes)
    if boxes is not None and len(boxes) > 0:
        plot = draw_bounding_boxes(plot, boxes)
    if points is not None and len(points) > 0:
        plot = draw_keypoints(
            plot,
            keypoints=points,
            colors="red",
        )
    return plot


@torch.no_grad()
def batch_draw(
    image_batch: Tensor,
    box_batch: Optional[list[Tensor]] = None,
    label_batch: Optional[list[Tensor]] = None,
    point_batch: Optional[list[Tensor]] = None,
) -> Tensor:
    empty_list = [None for _ in range(len(image_batch))]
    return make_grid(
        [
            draw(image=image, boxes=boxes, points=points)
            for image, boxes, points in zip(
                image_batch, box_batch or empty_list, point_batch or empty_list
            )
        ]
    )
