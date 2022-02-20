import torch

from vision_tools.metric import BoxAP


def test_boxap() -> None:
    ap = BoxAP()
    pred_boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    pred_confs = torch.tensor([0.9, 0.8, 0.7])
    gt_boxes = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
    ap.update([pred_boxes], [pred_confs], [gt_boxes])
    assert ap.value[0] == 0.5
