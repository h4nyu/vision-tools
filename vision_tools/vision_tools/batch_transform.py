import torch
from .interface import TrainBatch
from typing import List, Tuple, Dict, Any
from torchvision.ops import clip_boxes_to_image
import torch.nn.functional as F
from vision_tools.box import shift
import random


# TODO WIP
class BatchMosaic:
    def __init__(self) -> None:
        ...

    def __call__(self, batch: TrainBatch) -> TrainBatch:
        image_batch = batch["image_batch"]
        box_batch = batch["box_batch"]
        label_batch = batch["label_batch"]
        batch_size, _, image_height, image_width = image_batch.shape
        cross_x, cross_y = random.randint(0, image_width), random.randint(
            0, image_height
        )
        split_keys = [
            (0, 0, cross_x, cross_y),
            (cross_x, 0, image_width, cross_y),
            (0, cross_y, cross_x, image_height),
            (cross_x, cross_y, image_width, image_height),
        ]
        splited_image_batch, splited_box_batch, splited_label_batch = {}, {}, {}

        # split to 2x2 patch
        for key in split_keys:
            x0, y0, x1, y1 = key
            splited_image_batch[key] = image_batch[
                :, :, key[1] : key[3], key[0] : key[2]
            ]
            in_area = (
                lambda boxes: (x0 < boxes[:, 0])
                & (boxes[:, 0] < x1)
                & (y0 < boxes[:, 1])
                & (boxes[:, 1] < y1)
            )
            clip_boxes_to_image
            masks = [in_area(boxes) for boxes in box_batch]
            splited_box_batch[key] = [
                b[m].clamp(
                    min=torch.tensor([x0, y0, x0, y0]),
                    max=torch.tensor([x1, y1, x1, y1]),
                )
                for b, m in zip(box_batch, masks)
            ]

            splited_label_batch[key] = [l[m] for l, m in zip(label_batch, masks)]
        # concat
        return batch


class BatchRemovePadding:
    def __init__(self, original_size: Tuple[int, int]) -> None:
        self.original_size = original_size
        self.original_width, self.original_height  = original_size
        self.longest_side = max(self.original_width, self.original_height)

    def scale_and_pad(
        self, padded_size: Tuple[int, int]
    ) -> Tuple[float, Tuple[int, int]]:
        padded_width, padded_height = padded_size
        if self.longest_side == self.original_width:
            scale = self.longest_side / padded_width
            pad = (padded_height - self.original_height / scale) / 2
            return scale, (0, int(pad))
        else:
            scale = self.longest_side / padded_height
            pad = (padded_width - self.original_width / scale) / 2
            return scale, (int(pad), 0)

    @torch.no_grad()
    def __call__(self, batch: TrainBatch) -> TrainBatch:
        _image_batch = []
        box_batch = []
        for img, boxes in zip(batch["image_batch"], batch["box_batch"]):
            padded_size = (img.shape[2], img.shape[1])
            scale, pad = self.scale_and_pad(padded_size)
            boxes = shift(boxes, (-pad[0], -pad[1]))
            img = img[
                :, pad[1] : padded_size[1] - pad[1], pad[0] : padded_size[0] - pad[0]
            ]
            boxes = boxes * scale
            _image_batch.append(img)
            box_batch.append(boxes)
        image_batch = F.interpolate(
            torch.stack(_image_batch, dim=0),
            size=(self.original_height, self.original_width),
        )
        return {
            **batch,
            "image_batch": image_batch,
            "box_batch": box_batch,
        }
