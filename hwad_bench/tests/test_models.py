from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from hwad_bench.data import Submission
from hwad_bench.models import (
    EmbNet,
    EnsembleSubmission,
    save_submission,
    search_threshold,
)


def test_save_submission() -> None:
    submissions: list[Any] = [
        dict(
            image_file="img0-xxxxx.jpg",
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
        )
    ]
    rows = save_submission(submissions, "/app/test_outputs/submissions.csv")
    assert len(rows) == 1
    assert rows[0]["image"] == "img0.jpg"


def test_model() -> None:
    embedding_size = 512
    model = EmbNet(name="convnext_tiny", embedding_size=embedding_size)
    images = torch.randn(2, 3, 256, 256)
    output = model(images)
    assert output.shape == (2, embedding_size)


def test_ensemble_mean() -> None:
    fn = EnsembleSubmission()

    rows = [
        Submission(
            image_id="img0",
            individual_ids=["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"],
            distances=[0.5, 0.4, 0.3, 0.2, 0.1],
        ),
        Submission(
            image_id="img0",
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
            distances=[0.5, 0.4, 0.3, 0.2, 0.1],
        ),
    ]
    res = fn(rows)[0]
    assert res["image_id"] == "img0"
    assert res["individual_ids"] == ["tr-0", "tr-2", "tr-1", "tr-3", "tr-4"]
    assert res["distances"] == [0.5, 0.3, 0.28, 0.2, 0.1]


def test_ensemble_max() -> None:
    fn = EnsembleSubmission(strategy="max")

    rows = [
        Submission(
            image_id="img0",
            individual_ids=["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"],
            distances=[0.6, 0.4, 0.3, 0.2, 0.1],
        ),
        Submission(
            image_id="img0",
            individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
            distances=[0.5, 0.4, 0.3, 0.2, 0.1],
        ),
    ]
    res = fn(rows)[0]
    assert res["image_id"] == "img0"
    assert res["individual_ids"] == ["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"]
    assert res["distances"] == [0.6, 0.4, 0.3, 0.2, 0.1]
