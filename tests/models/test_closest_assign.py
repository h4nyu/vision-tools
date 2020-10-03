import torch
from object_detection.entities import YoloBoxes
from object_detection.models.closest_assign import ClosestAssign


def test_closest_assign() -> None:
    anchor = YoloBoxes(
        torch.tensor(
            [
                [0.11, 0.12, 0.1, 0.1],
                [0.21, 0.22, 0.1, 0.1],
                [0.25, 0.22, 0.1, 0.1],
                [0.31, 0.32, 0.1, 0.1],
                [0.41, 0.42, 0.1, 0.1],
                [0.61, 0.62, 0.1, 0.1],
            ]
        )
    )

    gt = YoloBoxes(
        torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.1, 0.1],
                [0.3, 0.3, 0.1, 0.1],
            ]
        )
    )
    fn = ClosestAssign(topk=2)
    matched_ids = fn(anchor, gt)
    assert matched_ids.tolist() == [[0, 1, 3], [1, 2, 2]]
