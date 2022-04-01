from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from hwad_bench.models import EmbNet, save_submission, search_threshold


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


# def test_search_thresold() -> None:
#     val_rows: list[Any] = [
#         {
#             "image_file": "img0-box0.png",
#             "species": "species-0",
#             "individual_id": "val-0",
#             "label": 1,
#         },
#         {
#             "image_file": "img1--b0.png",
#             "species": "species-0",
#             "individual_id": "tr-0",
#             "label": 2,
#         },
#     ]
#     train_rows: list[Any] = [
#         {
#             "image_file": "img0-box0.png",
#             "species": "species-0",
#             "individual_id": "tr-0",
#             "label": 1,
#         },
#         {
#             "image_file": "img1--b0.png",
#             "species": "species-0",
#             "individual_id": "tr-1",
#             "label": 2,
#         },
#     ]
#     submissions: list[Any] = [
#         dict(
#             image_file="img0-box0.png",
#             distances=[0.9, 0.8, 0.7, 0.6, 0.5],
#             individual_ids=["tr-0", "tr-1", "tr-1", "tr-1", "tr-1"],
#         ),
#     ]

#     res = search_threshold(
#         train_rows=train_rows,
#         val_rows=val_rows,
#         submissions=submissions,
#         thresholds=[0.5, 0.95],
#     )
#     assert res == [{"threshold": 0.5, "score": 0.0}, {"threshold": 0.95, "score": 1.0}]
