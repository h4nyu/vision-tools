import torch
from app.models.matcher import HungarianMatcher, Outputs, Targets


def test_hungarian_matcher() -> None:
    batch_size = 2
    num_queries = 20
    num_classes = 2
    outputs: Outputs = {
        "pred_logits": torch.rand(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    }
    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)

    targets: Targets = [
        {
            "labels": torch.ones((9,)),
            "boxes": torch.cat([torch.ones((9, 2)), torch.ones((9, 2)) * 1], dim=1),
        },
        {
            "labels": torch.zeros((2,)),
            "boxes": torch.cat([torch.ones((2, 2)), torch.ones((2, 2)) * 1], dim=1),
        },
    ]

    fn = HungarianMatcher()
    res = fn(outputs, targets)
    assert len(res) == batch_size
    for (pred_ids, tgt_ids), target in zip(res, targets):
        assert pred_ids.shape == (min(num_queries, len(target["labels"])),)
        assert tgt_ids.shape == pred_ids.shape
        assert max(pred_ids) <= num_queries
        assert max(tgt_ids) <= len(target["labels"])
