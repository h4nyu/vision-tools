import random
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch import Tensor, nn
from torch.nn.functional import interpolate
from torchvision.ops import box_convert
from torchvision.utils import (
    draw_bounding_boxes,
    draw_segmentation_masks,
    make_grid,
    save_image,
)
from typing_extensions import Protocol

from vision_tools import Number, resize_points

from .interface import TrainBatch

logger = getLogger(__name__)


def seed_everything(seed: int = 3801) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# wait torchvision 0.12 release
@torch.no_grad()
def draw_keypoints(
    image: Tensor,
    keypoints: Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> Tensor:

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


# wait torchvision 0.11 release in kaggle
def masks_to_boxes(masks: Tensor) -> Tensor:
    """
    Compute the bounding boxes around the provided masks.
    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.
    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(path: Union[str, Path], config: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f)


class ToDevice:
    def __init__(
        self,
        device: str,
    ) -> None:
        self.device = device

    def __call__(
        self, *args: Union[Tensor, List[Tensor]], **kargs: Union[Tensor, List[Tensor]]
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


class Checkpoint:
    def __init__(
        self,
        root_dir: str,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True, parents=True)

    def load(
        self,
        target: str = "best",
    ) -> Optional[dict]:
        path = self.root_dir / f"{target}.pth"
        if path.exists():
            state = torch.load(path)
            return state
        return None

    def save(
        self,
        state: dict,
        target: str = "best",
    ) -> None:
        path = self.root_dir / f"{target}.pth"
        torch.save(state, path)  # type: ignore


@torch.no_grad()
def draw(
    image: Tensor,
    masks: Optional[Tensor] = None,
    boxes: Optional[Tensor] = None,
    points: Optional[Tensor] = None,
    gt_boxes: Optional[Tensor] = None,
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
    if gt_boxes is not None and len(gt_boxes) > 0:
        plot = draw_bounding_boxes(plot, boxes=gt_boxes, colors=["red"] * len(gt_boxes))
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
    box_batch: Optional[List[Tensor]] = None,
    label_batch: Optional[List[Tensor]] = None,
    point_batch: Optional[List[Tensor]] = None,
    gt_box_batch: Optional[List[Tensor]] = None,
) -> Tensor:
    empty_list = [None for _ in range(len(image_batch))]
    return make_grid(
        [
            draw(image=image, boxes=boxes, points=points, gt_boxes=gt_boxes)
            for image, boxes, points, gt_boxes in zip(
                image_batch,
                box_batch or empty_list,
                point_batch or empty_list,
                gt_box_batch or empty_list,
            )
        ]
    )


def merge_batch(batches: List[TrainBatch]) -> TrainBatch:
    image_batch = torch.cat([batch["image_batch"] for batch in batches])
    box_batch, label_batch, conf_batch = [], [], []
    for batch in batches:
        box_batch.extend(batch["box_batch"])
        label_batch.extend(batch["label_batch"])
        conf_batch.extend(batch["conf_batch"])
    return {
        "image_batch": image_batch,
        "box_batch": box_batch,
        "label_batch": label_batch,
        "conf_batch": conf_batch,
    }
