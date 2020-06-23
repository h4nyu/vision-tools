import torch
from object_detection.models.set_criterion import SetCriterion
from object_detection.models.matcher import (
    HungarianMatcher,
    Outputs,
    Targets,
    MatchIndecies,
)


def test_loss_lables() -> None:
    src_logits = torch.tensor([[[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]]])  # 1, 0, 0

    assert src_logits.shape == (1, 3, 2)
    patters = [
        ([0, 0], [0, 1], [0, 1], 0.95),
        ([0, 0], [1, 2], [0, 1], 0.4),
        ([0], [2], [0], 0.65),
        ([0], [0], [0], 1.2),
    ]
    fn = SetCriterion(num_classes=1,)
    for tgt, src_ids, tgt_ids, loss in patters:
        tgt_labels = [torch.tensor(tgt).long()]  # dynamic size for each batch
        indices: MatchIndecies = [(torch.tensor(src_ids), torch.tensor(tgt_ids))]
        assert len(tgt_labels) == 1
        assert len(indices) == 1
        res = fn.loss_labels(src_logits, tgt_labels, indices)
        assert res < loss
