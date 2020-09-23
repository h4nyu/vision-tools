import torch
from object_detection.entities import (
    YoloBoxBatch,
    ConfidenceBatch,
)
from object_detection.models.matcher import HungarianMatcher


def test_hungarian_matcher() -> None:
    num_queries = 3
    num_classes = 2
    pred_box_batch = YoloBoxBatch(
        torch.tensor(
            [
                [
                    [0.1, 0.1, 0.2, 0.2],
                    [0.2, 0.4, 0.1, 0.1],
                    [0.2, 0.4, 0.1, 0.1],
                ]
            ]
        )
    )
    pred_cfd_batch = ConfidenceBatch(
        torch.tensor(
            [
                [
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.5, 0.5],
                ]
            ]
        )
    )


#      targets: Targets = [
#          {
#              "labels": torch.tensor([1, 0]).long(),
#              "boxes": torch.tensor([[3, 3, 1, 1], [2, 2, 1, 1]]).float(),
#          },
#      ]
#      assert len(targets) == 1
#      assert targets[0]["boxes"].shape == (2, 4)
#      assert targets[0]["labels"].shape == (2,)
#
#      cost_patterns = [
#          ((1, 0, 0), ([0, 1], [0, 1])),
#          ((0, 1, 0), ([1, 2], [1, 0])),
#          ((0, 0, 1), ([1, 2], [1, 0])),
#          ((1, 1, 1), ([1, 2], [1, 0])),
#      ]
#      for (cost_class, cost_box, cost_giou), (src_ids, tgt_ids) in cost_patterns:
#          fn = HungarianMatcher(
#              cost_class=cost_class, cost_box=cost_box, cost_giou=cost_giou,
#          )
#          res = fn(outputs, targets)
#          assert len(res) == 1
#          assert (res[0][0] == torch.tensor(src_ids)).all()
#          assert (res[0][1] == torch.tensor(tgt_ids)).all()
