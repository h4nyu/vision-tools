from __future__ import annotations

import pytest
import torch

from hwad_bench.metrics import MeanAveragePrecisionK


@pytest.mark.parametrize(
    "pred,expected",
    [
        ([1, 2, 3, 4], 1.0),
        ([2, 1, 3, 4], 0.5),
        ([2, 3, 4, 1], 0.25),
        ([2, 3, 4, 5], 0.0),
    ],
)
def test_mean_average_precision_k(pred: list[int], expected: float) -> None:
    m = MeanAveragePrecisionK()
    gt_labels = torch.tensor([1]).long()
    pred_labels = torch.tensor([pred]).long()
    m.update(pred_labels, gt_labels)
    score, _ = m.value
    assert score == pytest.approx(expected)


def test_mean_average_precision_k_shape() -> None:
    m = MeanAveragePrecisionK()
    gt_labels = torch.tensor([1, 1]).long()
    pred_labels = torch.ones((2, 5)).long()
    m.update(pred_labels, gt_labels)
