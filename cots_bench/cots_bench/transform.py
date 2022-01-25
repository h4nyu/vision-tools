import torch
from torch import Tensor, nn
from typing import Dict, Tuple, Any, Optional
from vision_tools.interface import TrainSample
from torchvision.ops import box_convert
from vision_tools.box import resize_boxes
import random
import torch.nn.functional as F


class RandomCutAndPaste:
    def __init__(
        self,
        radius: float = 0.3,
        mask_size: int = 64,
        use_vflip: bool = False,
        use_hflip: bool = False,
        use_rot90: bool = False,
        scale_limit: Optional[Tuple[float, float]] = None,
        p: float = 1.0,
    ) -> None:
        self.radius = radius
        self.mask = self._make_mask(mask_size)
        self.use_vflip = use_vflip
        self.use_hflip = use_hflip
        self.use_rot90 = use_rot90
        self.scale_limit = scale_limit
        self.p = p

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

        labels = sample["labels"]
        confs = sample["confs"]

        device = image.device
        _, H, W = image.shape
        paste_image = image.clone()
        cxcywhs = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

        radius = int(self.radius * min(H, W))
        for idx in range(box_count):
            if random.uniform(0, 1) > self.p:
                continue

            cut_box = boxes[idx]
            cut_label = sample["labels"][idx]
            cut_conf = sample["confs"][idx]
            cut_cxcywh = cxcywhs[idx]
            paste_width = int(cut_cxcywh[2])
            paste_height = int(cut_cxcywh[3])
            scale = (
                1.0
                if self.scale_limit is None
                else random.uniform(self.scale_limit[0], self.scale_limit[1])
            )
            # guard against too small boxes
            if (paste_width * scale <= 1.0) or (paste_height * scale <= 1.0):
                continue

            max_length = int(max(paste_width, paste_height) * scale)

            paste_x0 = torch.randint(
                int(cut_box[0] - radius), int(cut_box[0] + radius), (1,)
            ).clamp(min=0, max=W - max_length)
            paste_y0 = torch.randint(
                int(cut_box[1] - radius), int(cut_box[1] + radius), (1,)
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

            boxes = torch.cat([boxes, paste_box.unsqueeze(0)])
            labels = torch.cat([labels, cut_label.unsqueeze(0)])
            confs = torch.cat([confs, cut_conf.unsqueeze(0)])

        return {
            "image": paste_image,
            "boxes": boxes,
            "labels": labels,
            "confs": confs,
        }


class FilterSmallBoxes:
    def __init__(
        self,
        min_height: int = 4,
        min_width: int = 4,
        aspect_limit: float = 3,
    ) -> None:
        self.min_height = min_height
        self.min_width = min_width
        self.aspect_limit = aspect_limit

    def __call__(self, sample: TrainSample) -> TrainSample:
        image = sample["image"]
        boxes = sample["boxes"]
        box_count = boxes.shape[0]
        if box_count == 0:
            return sample

        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        filter_mask = (box_widths >= self.min_width) & (box_heights >= self.min_height)
        aspects, _ = torch.stack(
            [
                box_widths.clamp(min=1.0) / box_heights.clamp(min=1.0),
                box_heights.clamp(min=1.0) / box_widths.clamp(min=1.0),
            ]
        ).max(dim=0)
        filter_mask = filter_mask & (aspects <= self.aspect_limit)
        return {
            "image": image,
            "boxes": boxes[filter_mask],
            "labels": sample["labels"][filter_mask],
            "confs": sample["confs"][filter_mask],
        }


class Resize:
    def __init__(
        self,
        height: int,
        width: int,
    ) -> None:
        self.height = height
        self.width = width

    def __call__(self, sample: TrainSample) -> TrainSample:
        image = sample["image"]
        boxes = sample["boxes"]
        _, H, W = image.shape
        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                self.height,
                self.width,
            ),
            mode="nearest",
        )[0]
        boxes = resize_boxes(boxes, (self.width / W, self.height / H))
        return {
            "image": image,
            "boxes": boxes,
            "labels": sample["labels"],
            "confs": sample["confs"],
        }
