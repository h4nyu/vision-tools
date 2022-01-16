import pytest
import torch
from vision_tools.batch_transform import BatchMosaic, RemovePadding
from vision_tools.interface import TrainBatch


@pytest.fixture
def batch() -> TrainBatch:
    image_batch = torch.rand(2, 3, 32, 24)
    box_batch = [
        torch.tensor(
            [
                [2, 2, 3, 3],
                [22, 30, 24, 32],
            ]
        ).float(),
        torch.tensor([[2, 2, 3, 3]]).float(),
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

    return TrainBatch(
        image_batch=image_batch,
        box_batch=box_batch,
        label_batch=label_batch,
    )


def test_batch_mosaic(batch: TrainBatch) -> None:
    t = BatchMosaic()
    t(batch)


def test_remove_padding(batch: TrainBatch) -> None:
    orignal_size = (720, 1280)
    t = RemovePadding(orignal_size)
    t(batch)
