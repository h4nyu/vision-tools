import torch
from torch import Tensor, nn
from typing import Dict, Tuple, Any
from vision_tools.interface import TrainSample
from torchvision.ops import box_convert
import torch.nn.functional as F
import numpy as np


class RandomCutAndPaste:
    def __init__(
        self,
        radius: int = 128,
        mask_size: int = 64,
    ) -> None:
        self.radius = radius
        self.mask = self._make_mask(mask_size)

    @torch.no_grad()
    def _make_mask(self, mask_size: int) -> Tensor:
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(mask_size, dtype=torch.int64),
            torch.arange(mask_size, dtype=torch.int64),
        )
        grid_xy = torch.stack([grid_x, grid_y])
        cxcy = torch.tensor([mask_size / 2, mask_size / 2]).view(2, 1, 1).int()
        mask = (mask_size / 2) ** 2 - ((grid_xy - cxcy) ** 2).sum(dim=0)
        mask = (
            mask.float().view(1, mask_size, mask_size).unsqueeze(0).clamp(min=0, max=1)
        )
        mask = F.avg_pool2d(
            mask, kernel_size=mask_size // 2, padding=mask_size // 2 // 2, stride=1
        )
        return mask

    def __call__(self, sample: TrainSample) -> TrainSample:
        image = sample["image"]
        boxes = sample["boxes"]
        if len(boxes) == 0:
            return sample

        paste_image = image.clone()
        cxcywhs = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        _, H, W = image.shape
        u = np.random.choice(len(boxes))
        cut_box = boxes[u].int()
        cut_cxcywh = cxcywhs[u]

        paste_x0y0 = cut_cxcywh[:2] + np.random.randint(
            -self.radius, self.radius, size=2
        )
        paste_wh = cut_cxcywh[2:]
        paste_box = torch.cat([paste_x0y0, paste_x0y0 + paste_wh]).int()

        blend_mask = F.interpolate(
            self.mask,
            size=(paste_box[3] - paste_box[1], paste_box[2] - paste_box[0]),
            mode="nearest",
        )[0]
        paste_image[:, int(paste_box[1]) : int(paste_box[3]), int(paste_box[0]) : int(paste_box[2])] = (
            blend_mask * image[:, int(cut_box[1]) : int(cut_box[3]), int(cut_box[0]) : int(cut_box[2])]
            + (1 - blend_mask)
            * image[:, int(paste_box[1]) : int(paste_box[3]), int(paste_box[0]) : int(paste_box[2])]
        )

        pasted_boxes = torch.cat([boxes, paste_box.unsqueeze(0)])

        return {
            "image": paste_image,
            "boxes": pasted_boxes,
            "labels": sample["labels"],
            "confs": sample["confs"],
        }
