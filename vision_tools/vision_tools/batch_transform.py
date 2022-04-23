import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_area, clip_boxes_to_image

from vision_tools.box import filter_aspect, shift

from .interface import TrainBatch


class BatchRelocate:
    def __init__(self, n_copy: int = 1) -> None:
        self.n_copy = n_copy

    def _relocate(
        self, image: Tensor, boxes: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_boxes = boxes.shape[0]
        if n_boxes == 0:
            return image, boxes, labels

        box_sizes = boxes[:, 2:] - boxes[:, :2]
        max_wh, _ = box_sizes.max(dim=0)
        _, h, w = image.shape
        device = image.device
        image_size = torch.tensor([h, w]).int().to(device)
        point_matrix = torch.cat(
            [
                torch.randint(
                    0,
                    int((w - max_wh[0]).clamp(min=1)),
                    (
                        n_boxes,
                        self.n_copy,
                        1,
                    ),
                    device=device,
                ),
                torch.randint(
                    0,
                    int((h - max_wh[1]).clamp(min=1)),
                    (
                        n_boxes,
                        self.n_copy,
                        1,
                    ),
                    device=device,
                ),
            ],
            dim=2,
        )
        new_box_list, new_label_list = [boxes], [labels]
        for b, points, label in zip(boxes.int(), point_matrix, labels):
            box_image = image[:, b[1] : b[3], b[0] : b[2]]
            _, box_h, box_w = box_image.shape
            new_boxes = torch.cat(
                [
                    points,
                    points[:, 0:1] + box_w,
                    points[:, 1:2] + box_h,
                ],
                dim=1,
            )
            new_boxes = clip_boxes_to_image(new_boxes, [h, w])
            for nb in new_boxes:
                image[:, nb[1] : nb[3], nb[0] : nb[2]] = box_image
            new_box_list.append(new_boxes)
            new_label_list.append(torch.full((self.n_copy,), label).to(label))
        boxes = torch.cat(new_box_list, dim=0)
        labels = torch.cat(new_label_list, dim=0)
        return image, boxes, labels

    @torch.no_grad()
    def __call__(self, batch: TrainBatch) -> TrainBatch:
        image_batch = batch["image_batch"]
        device = image_batch.device
        box_batch = batch["box_batch"]
        label_batch = batch["label_batch"]
        for i, (image, boxes, labels) in enumerate(
            zip(image_batch, box_batch, label_batch)
        ):
            image, boxes, labels = self._relocate(image, boxes, labels)
            image_batch[i] = image
            box_batch[i] = boxes
            label_batch[i] = labels
        return batch


class BatchMosaic:
    def __init__(
        self,
        width_limit: Tuple[float, float] = (1 / 5, 4 / 5),
        height_limit: Tuple[float, float] = (1 / 5, 4 / 5),
        aspect_limit: float = 2.0,
        p: float = 0.5,
    ) -> None:
        self.width_limit = width_limit
        self.height_limit = height_limit
        self.aspect_limit = aspect_limit
        self.p = p

    @torch.no_grad()
    def __call__(self, batch: TrainBatch) -> TrainBatch:
        if random.uniform(0, 1) > self.p:
            return batch
        image_batch = batch["image_batch"]
        device = image_batch.device
        box_batch = batch["box_batch"]
        label_batch = batch["label_batch"]
        batch_size, _, image_height, image_width = image_batch.shape
        cross_x, cross_y = random.randint(
            int(image_width * self.width_limit[0]),
            int(image_width * self.width_limit[1]),
        ), random.randint(
            int(image_height * self.height_limit[0]),
            int(image_height * self.height_limit[1]),
        )
        split_keys = [
            (0, 0, cross_x, cross_y),
            (cross_x, 0, image_width, cross_y),
            (0, cross_y, cross_x, image_height),
            (cross_x, cross_y, image_width, image_height),
        ]
        splited_image_batch, splited_box_batch, splited_label_batch, recipe_batch = (
            {},
            {},
            {},
            {},
        )
        recipe_index = list(range(batch_size))
        # split to 2x2 patch
        for key in split_keys:
            x0, y0, x1, y1 = key
            splited_image_batch[key] = image_batch[:, :, y0:y1, x0:x1]
            min_area = torch.tensor([x0, y0, x0, y0]).to(device)
            max_area = torch.tensor([x1, y1, x1, y1]).to(device)
            cropped_box_batch = [
                boxes.clamp(
                    min=min_area,
                    max=max_area,
                )
                if len(boxes) > 0
                else boxes
                for boxes in box_batch
            ]
            in_area_batch = [
                (box_area(boxes) > 0) & (filter_aspect(boxes, self.aspect_limit))
                if len(boxes) > 0
                else torch.zeros(0).bool().to(device)
                for boxes in cropped_box_batch
            ]

            splited_box_batch[key] = [
                b[m] for b, m in zip(cropped_box_batch, in_area_batch)
            ]

            splited_label_batch[key] = [
                l[m] for l, m in zip(label_batch, in_area_batch)
            ]
            random.shuffle(recipe_index)
            recipe_batch[key] = recipe_index.copy()

        for key, recipe_index in recipe_batch.items():
            x0, y0, x1, y1 = key
            image_batch[:, :, y0:y1, x0:x1] = splited_image_batch[key][recipe_index]
            splited_box_batch[key] = [splited_box_batch[key][i] for i in recipe_index]
            splited_label_batch[key] = [
                splited_label_batch[key][i] for i in recipe_index
            ]
        for i in range(batch_size):
            box_batch[i] = torch.cat([splited_box_batch[key][i] for key in split_keys])
            label_batch[i] = torch.cat(
                [splited_label_batch[key][i] for key in split_keys]
            )

        batch["image_batch"] = image_batch
        batch["box_batch"] = box_batch
        batch["label_batch"] = label_batch
        return batch


class BatchRemovePadding:
    def __init__(self, original_size: Tuple[int, int]) -> None:
        self.original_size = original_size
        self.original_width, self.original_height = original_size
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
        out_batch: Any = {
            **batch,
            "image_batch": image_batch,
            "box_batch": box_batch,
        }
        return out_batch
