from app.eval import MeamPrecition, precition, Evaluate
import torch


def test_meanap() -> None:
    fn = MeamPrecition()
    pred_boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1],])

    gt_boxes = torch.tensor(
        [[0.0, 0.0, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3],]
    )
    fn(pred_boxes, gt_boxes)


def test_precition() -> None:
    iou_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.6],
            [0.9, 1.0, 0.7],
            [0.9, 1.0, 0.7],
            [0.0, 0.3, 0.0,],
            [0.1, 0.2, 0.1,],
            [0.1, 0.2, 0.1,],
        ],
        dtype=torch.float32,
    )
    confidences = torch.tensor([0.6, 0.6, 0.6, 0.5, 0.7, 0.8, 0.9])
    res = precition(iou_matrix=iou_matrix, threshold=0.5)
    assert res == 1 / (1 + 3 + 2)
