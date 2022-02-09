from vision_tools.tracker import Tracker
import torch
from vision_tools.interface import TrainSample


def test_tracker() -> None:
    sample0 = TrainSample(
        image=torch.rand(3, 224, 224),
        boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([0]),
        confs=torch.tensor([1.0]),
    )
    sample1 = TrainSample(
        image=torch.rand(3, 224, 224),
        boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([0]),
        confs=torch.tensor([1.0]),
    )
    tracker = Tracker()

    tracker(sample0)
    tracker(sample1)
