import torch
from torch import Tensor, nn
from typing import Dict, Tuple, Any, Optional
from vision_tools.interface import TrainSample
from torchvision.ops import box_convert
import random
import torch.nn.functional as F


class RandomCutAndPaste:
    def __init__(
        self,
        radius: int = 256,
        mask_size: int = 64,
        use_vflip: bool = False,
        use_hflip: bool = False,
        use_rot90: bool = False,
        scale_limit: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.radius = radius
        self.mask = self._make_mask(mask_size)
        self.use_vflip = use_vflip
        self.use_hflip = use_hflip
        self.use_rot90 = use_rot90
        self.scale_limit = scale_limit

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
        box_count = boxes.shape[0]
        if box_count == 0:
            return sample

        device = image.device
        _, H, W = image.shape
        paste_image = image.clone()
        u = random.randint(0, box_count - 1)
        cut_box = boxes[u]
        cut_label = sample["labels"][u]
        cut_conf = sample["confs"][u]
        cut_cxcywh = box_convert(boxes[u : u + 1], in_fmt="xyxy", out_fmt="cxcywh")[0]
        paste_width = int(cut_cxcywh[2])
        paste_height = int(cut_cxcywh[3])

        # guard against too small boxes
        if (paste_width * scale <= 1.0) or (paste_height * scale <= 1.0):
            return sample

        scale = (
            1.0
            if self.scale_limit is None
            else random.uniform(self.scale_limit[0], self.scale_limit[1])
        )
        max_length = int(max(paste_width, paste_height) * scale)

        paste_x0 = torch.randint(
            int(cut_box[0] - self.radius), int(cut_box[0] + self.radius), (1,)
        ).clamp(min=0, max=W - max_length)
        paste_y0 = torch.randint(
            int(cut_box[1] - self.radius), int(cut_box[1] + self.radius), (1,)
        ).clamp(min=0, max=H - max_length)

        paste_box = torch.cat(
            [paste_x0, paste_y0, paste_x0 + paste_width, paste_y0 + paste_height]
        )

        blend_mask = F.interpolate(
            self.mask,
            size=(paste_height, paste_width),
            mode="nearest",
        )[0]
        box_image = image[
            :,
            int(cut_box[1]) : int(cut_box[1]) + paste_height,
            int(cut_box[0]) : int(cut_box[0]) + paste_width,
        ]

        if self.use_vflip and random.uniform(0, 1) > 0.5:
            box_image = box_image.flip(1)
        if self.use_hflip and random.uniform(0, 1) > 0.5:
            box_image = box_image.flip(2)
        if self.use_rot90 and random.uniform(0, 1) > 0.5:
            box_image = torch.rot90(box_image, dims=[1, 2])
            blend_mask = torch.rot90(blend_mask, dims=[1, 2])
            paste_width, paste_height = paste_height, paste_width
            paste_box[2] = paste_x0 + paste_width
            paste_box[3] = paste_y0 + paste_height

        if self.scale_limit is not None:
            paste_height, paste_width = int(paste_height * scale), int(
                paste_width * scale
            )
            blend_mask = F.interpolate(
                blend_mask.unsqueeze(0),
                size=(paste_height, paste_width),
                mode="nearest",
            )[0]
            box_image = F.interpolate(
                box_image.unsqueeze(0),
                size=(paste_height, paste_width),
                mode="nearest",
            )[0]
            paste_box[2] = paste_x0 + paste_width
            paste_box[3] = paste_y0 + paste_height

        paste_image[
            :,
            int(paste_box[1]) : int(paste_box[1]) + paste_height,
            int(paste_box[0]) : int(paste_box[0]) + paste_width,
        ] = (
            blend_mask * box_image
            + (1 - blend_mask)
            * image[
                :,
                int(paste_box[1]) : int(paste_box[1]) + paste_height,
                int(paste_box[0]) : int(paste_box[0]) + paste_width,
            ]
        )

        paste_boxes = torch.cat([boxes, paste_box.unsqueeze(0)])
        paste_labels = torch.cat([sample["labels"], cut_label.unsqueeze(0)])
        paste_confs = torch.cat([sample["confs"], cut_conf.unsqueeze(0)])

        return {
            "image": paste_image,
            "boxes": paste_boxes,
            "labels": paste_labels,
            "confs": paste_confs,
        }
