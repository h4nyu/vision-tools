import torch
from vision_tools.batch_transform import BatchMosaic
from vision_tools.interface import TrainBatch


def test_batch_mosaic() -> None:
    t = BatchMosaic()
    image_batch = torch.rand(2, 3, 32, 24)
    box_batch = [
        torch.tensor(
            [
                [2, 2, 3, 3],
                [22, 30, 24, 32],
            ]
        ),
        torch.tensor([[2, 2, 3, 3]]),
    ]
    label_batch = [
        torch.tensor(
            [
                0,
                0,
            ]
        ),
        torch.tensor(
            [
                0,
            ]
        ),
    ]

    batch = TrainBatch(
        image_batch=image_batch,
        box_batch=box_batch,
        label_batch=label_batch,
    )
    t(batch)
