import torch
from typing import List, Tuple, Any
from object_detection.entities import (
    ImageBatch,
    YoloBoxes,
    PascalBoxes,
    Labels,
    yolo_to_pascal,
    pascal_to_yolo,
)


class Cutmix:
    def __init__(self) -> None:
        ...

    def __call__(
        self, images: ImageBatch, box_batch: List[YoloBoxes], label_batch: List[Labels]
    ) -> Tuple[ImageBatch, List[YoloBoxes], List[Labels]]:
        b, _, image_h, image_w = images.shape
        base_ids = torch.arange(b)
        idset = torch.stack([torch.roll(base_ids, i) for i in range(4)]).t()

        points = torch.stack(
            [
                torch.randint(low=0, high=image_w, size=(b,)),
                torch.randint(low=0, high=image_h, size=(b,)),
            ]
        ).t()

        out_box_batch = []
        for ids, point in zip(idset, points):
            images[ids[0], :, 0 : point[0], 0 : point[1]] = images[ids[1], :, 0 : point[0], 0 : point[1]]  # type: ignore
            pboxes0 = yolo_to_pascal(box_batch[int(ids[0])], (image_w, image_h))
            pboxes1 = yolo_to_pascal(box_batch[int(ids[1])], (image_w, image_h))
            mask1 = (pboxes1[:, 0] < point[0]) & (pboxes1[:, 1] < point[1])
            pboxes1 = PascalBoxes(pboxes1[mask1])
            pboxes1[:, 2] = pboxes1[:, 2].clamp(max=point[0].item())
            pboxes1[:, 3] = pboxes1[:, 3].clamp(max=point[1].item())
            boxes1 = pascal_to_yolo(pboxes1, (image_w, image_h))

            mask0 = (pboxes0[:, 0] < point[0]) & (pboxes0[:, 1] < point[1])
            pboxes0 = PascalBoxes(pboxes0[~mask0])
            boxes0 = pascal_to_yolo(pboxes0, (image_w, image_h))

            out_box_batch.append(YoloBoxes(torch.cat([boxes0, boxes1], dim=0)))
        #      t_img = images[tgt_id]
        #      print(s_img)
        #  area0 = torch.cat([
        #      torch.zeros(point.shape, dtype=torch.long),
        #      point
        #  ],dim=1)[:, [0,2,1,3]]
        #  area0 = images[:, :, area0[:, 1]:area0[:,3], area0[:, 0]:area0[:,2]] # type:ignore
        #
        #  print(area0)
        return images, out_box_batch, label_batch
