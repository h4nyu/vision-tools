import torch
from object_detection.entities import PascalBoxes
from object_detection.models.assign import ClosestAssign


def test_closest_assign() -> None:
    anchor = PascalBoxes(
        torch.tensor(
            [
                [5, 5, 6, 6],
                [10, 10, 11, 11],
                [25, 25, 26, 26],
                [32, 32, 33, 33],
                [41, 41, 42, 42],
                [61, 61, 62, 62],
            ]
        )
    )

    gt = PascalBoxes(
        torch.tensor(
            [
                [10, 10, 11, 11],
                [20, 20, 21, 21],
                [30, 30, 31, 31],
            ]
        )
    )
    fn = ClosestAssign(topk=2)
    matched_ids = fn(anchor, gt)
    assert matched_ids.tolist() == [[1, 0], [2, 1], [3, 2]]
