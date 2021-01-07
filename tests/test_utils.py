import torch
from object_detection.utils import DetectionPlot
from object_detection.entities.box import PascalBoxes, Labels, Confidences


def test_detection_plot() -> None:
    boxes = PascalBoxes(
        torch.tensor(
            [
                [10, 10, 300, 300],
            ]
        )
    )
    img = torch.ones((3, 1000, 1000), dtype=torch.uint8)
    labels = Labels(torch.tensor([10], dtype=torch.int32))
    confidences = Confidences(torch.tensor([0.5]))
    plot = DetectionPlot(img)
    plot.draw_boxes(
        boxes=boxes,
        labels=labels,
        confidences=confidences,
    )
    plot.save("/store/test-plot.jpg")
