import torch
from app.models.matcher import HungarianMatcher, Outputs, Targets


def test_hungarian_matcher() -> None:
    batch_size = 1
    num_queries = 2
    num_classes = 2
    outputs: Outputs = {
        "pred_logits": torch.tensor([[[0.5, 0.5], [0.1, 0.9],]]),
        "pred_boxes": torch.tensor([[[0, 0, 2, 2], [2, 2, 4, 4],],]).float(),
    }
    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)

    targets: Targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0, 0, 2, 2], [1, 1, 2, 2]]).float(),
        }
    ]

    fn = HungarianMatcher()
    fn(outputs, targets)
