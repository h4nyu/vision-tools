import torch
from .interface import TrainBatch
from torchvision.ops import clip_boxes_to_image
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
