import torch
from vision_tools.assign import ClosestAssign, SimOTA


def test_closest_assign() -> None:
    anchor = torch.tensor(
        [
            [5, 5, 6, 6],
            [10, 10, 11, 11],
            [25, 25, 26, 26],
            [32, 32, 33, 33],
            [41, 41, 42, 42],
            [61, 61, 62, 62],
        ]
    )

    gt = torch.tensor(
        [
            [10, 10, 11, 11],
            [20, 20, 21, 21],
            [30, 30, 31, 31],
        ]
    )
    fn = ClosestAssign(topk=2)
    matched_ids = fn(anchor, gt)
    assert matched_ids.tolist() == [[1, 0], [2, 1], [3, 2]]

def test_simota() -> None:
    anchor_points = torch.zeros(4, 2)
    anchor_points[0] = torch.tensor([5, 5])
    anchor_points[1] = torch.tensor([5, 5])
    anchor_points[2] = torch.tensor([5, 5])
    anchor_points[3] = torch.tensor([5, 5])

    pred_boxes = torch.zeros(4, 4)
    pred_boxes[0] = torch.tensor([5, 5, 8, 8])
    pred_boxes[1] = torch.tensor([5, 5, 8, 8])
    pred_boxes[2] = torch.tensor([5, 5, 8, 8])
    pred_boxes[3] = torch.tensor([5, 5, 8, 8])
    pred_scores = torch.tensor([0.8, 0.4, 0.8, 0.8])
    strides = torch.tensor([1.0, 2.0, 4.0, 8.0])

    gt_boxes = torch.zeros(1, 4)
    gt_boxes[0] = torch.tensor([3, 3, 9, 9])
    a = SimOTA(topk=6, radius=0.5)
    pair = a(anchor_points, pred_boxes, pred_scores, gt_boxes, strides)
    assert pair.tolist() == [
        [0, 2],
    ]
