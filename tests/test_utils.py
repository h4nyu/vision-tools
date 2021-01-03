import torch
from object_detection.utils import DetectionPlot
from object_detection.entities.box import PascalBoxes, Labels


def test_detection_plot() -> None:
    in_boxes = PascalBoxes(
        torch.tensor(
            [
                [10, 10, 30, 30],
            ]
        )
    )
    labels = Labels(torch.tensor([1], dtype=torch.int32))
    plot = DetectionPlot(w=100, h=100)
    plot.with_pascal_boxes(in_boxes, color="blue", labels=labels)
    plot.save(f"store/test-plot.png")
