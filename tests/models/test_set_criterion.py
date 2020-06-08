import torch
from app.models.set_criterion import SetCriterion
from app.models.matcher import HungarianMatcher, Outputs, Targets


def test_setlosses() -> None:
    batch_size = 2
    num_queries = 5
    num_classes = 2
    outputs: Outputs = {
        "pred_logits": torch.rand(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    }

    targets: Targets = [
        {
            "labels": torch.zeros((9,)).long(),
            "boxes": torch.cat([torch.ones((9, 2)), torch.ones((9, 2)) * 1], dim=1),
        },
        {
            "labels": torch.zeros((2,)).long(),
            "boxes": torch.cat([torch.ones((2, 2)), torch.ones((2, 2)) * 1], dim=1),
        },
    ]
    fn = SetCriterion(num_classes=num_classes, weights={},)
    res = fn.forward(outputs, targets)
    print(f"{res=}")


#  def tet_label_loss() -> None:
#      batch_size = 2
#      num_queries = 5
#      num_classes = 2
#      outputs: Outputs = {
#          "pred_logits": torch.rand(batch_size, num_queries, num_classes),
#          "pred_boxes": torch.rand(batch_size, num_queries, 4),
#      }
#
#      targets: Targets = [
#          {
#              "labels": torch.ones((9,)),
#              "boxes": torch.cat([torch.ones((9, 2)), torch.ones((9, 2)) * 1], dim=1),
#          },
#          {
#              "labels": torch.zeros((2,)),
#              "boxes": torch.cat([torch.ones((2, 2)), torch.ones((2, 2)) * 1], dim=1),
#          },
#      ]
#
#      fn = SetCriterion(num_classes=2)
#      #  fn.loss_labels(
#      #      outputs, targets, indices,
#      #  )


def test_forward() -> None:
    batch_size = 2
    num_queries = 5
    num_classes = 2
    outputs: Outputs = {
        "pred_logits": torch.rand(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    }

    targets: Targets = [
        {
            "labels": torch.zeros((9,)).long(),
            "boxes": torch.cat([torch.ones((9, 2)), torch.ones((9, 2)) * 1], dim=1),
        },
        {
            "labels": torch.zeros((2,)).long(),
            "boxes": torch.cat([torch.ones((2, 2)), torch.ones((2, 2)) * 1], dim=1),
        },
    ]
    fn = SetCriterion(num_classes=num_classes, weights={},)
    res = fn.forward(outputs, targets)
    print(f"{res=}")


#  def tet_label_loss() -> None:
#      batch_size = 2
#      num_queries = 5
#      num_classes = 2
#      outputs: Outputs = {
#          "pred_logits": torch.rand(batch_size, num_queries, num_classes),
#          "pred_boxes": torch.rand(batch_size, num_queries, 4),
#      }
#
#      targets: Targets = [
#          {
#              "labels": torch.ones((9,)),
#              "boxes": torch.cat([torch.ones((9, 2)), torch.ones((9, 2)) * 1], dim=1),
#          },
#          {
#              "labels": torch.zeros((2,)),
#              "boxes": torch.cat([torch.ones((2, 2)), torch.ones((2, 2)) * 1], dim=1),
#          },
#      ]
#
#      fn = SetCriterion(num_classes=2)
#      #  fn.loss_labels(
#      #      outputs, targets, indices,
#      #  )
