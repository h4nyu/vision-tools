from typing import Any

import torch
from predictor import Ensemble, MAPKMetric, Net


def ap(preds: Any, gts: Any, k: float) -> float:
    pred_count = 0
    dtc_count = 0
    num_positives = len(gts)
    score = 0.0
    for pred in preds:
        print(f"pred={pred}")
        pred_count += 1
        if pred in gts:
            dtc_count += 1
            score += dtc_count / pred_count
            gts.remove(pred)
        if len(gts) == 0:
            break
    score /= min(num_positives, k)
    return score


def test_net() -> None:
    net = Net(
        embedding_size=2,
        name="tf_efficientnet_b3_ns",
    )
    images = torch.randn(2, 3, 256, 256)
    out = net(images)


def test_ap() -> None:
    topk = 10
    preds = [
        4,
        1,
        2,
        3,
    ]
    gts = [
        3,
        2,
        1,
    ]
    score = ap(preds, [*gts], topk)
    metric = MAPKMetric(topk=topk)
    metric.reset()
    metric.update(torch.tensor(preds), torch.tensor(gts))
    assert metric.compute() == score


def test_ensemble() -> None:
    ensemble = Ensemble(topk=4)
    predictions = [
        ["a", "b", "c", "e"],
        ["b", "c", "a", "f"],
    ]
    merged = ensemble(predictions)
    assert merged == ["b", "a", "c", "e"]
