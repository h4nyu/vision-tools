import typing
import torch
from object_detection.entities import PascalBoxes
from object_detection.models.atss import ATSS


def test_atss_bench(benchmark: typing.Any) -> None:
    anchor = PascalBoxes(
        torch.tensor(
            [
                [11, 11, 21, 21],
                [21, 22, 32, 32],
                [25, 25, 35, 35],
                [35, 35, 45, 45],
            ]
        ).float()
    )

    gt = PascalBoxes(
        torch.tensor(
            [
                [10, 10, 20, 20],
                [20, 20, 30, 30],
            ]
        ).float()
    )
    fn = ATSS(topk=3)
    res = benchmark(lambda: fn(anchor, gt))
    assert res.tolist() == [[0, 0], [1, 1]]
