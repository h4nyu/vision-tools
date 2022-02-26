from __future__ import annotations

from typing import Any

from torch import Tensor


class MeanAveragePrecisionK:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.precisions: list[float] = []

    def update(self, labels_at_k: Tensor, gt_labels: Tensor) -> None:
        matching = labels_at_k == gt_labels.unsqueeze(1).expand(labels_at_k.shape)
        mask = matching.any(dim=1)
        matched_count = matching.int().argmax(dim=1) + 1
        precision = (1 / matched_count) * mask
        self.precisions += precision.tolist()

    @property
    def value(self) -> tuple[float, dict[str, Any]]:
        if len(self.precisions) == 0:
            return 0.0, {"precision": 0.0}
        precision = sum(self.precisions) / len(self.precisions)
        return precision, {"precision": precision}
