import torch
from vnet.utils import DetectionPlot
from vnet import Boxes, Labels, Confidences, Points, resize_points


def test_detection_plot() -> None:
    boxes = Boxes(
        torch.tensor(
            [
                [10, 10, 300, 300],
            ]
        )
    )
    points = Points(
        torch.tensor(
            [
                [30, 30],
            ]
        )
    )
    img = torch.ones((3, 1000, 1000), dtype=torch.uint8)
    labels = Labels(torch.tensor([10], dtype=torch.int32))
    confidences = Confidences(torch.tensor([0.5]))
    plot = DetectionPlot(img)
    plot.draw_points(
        points=points,
        labels=labels,
        confidences=confidences,
    )
    plot.draw_boxes(
        boxes=boxes,
        labels=labels,
        confidences=confidences,
    )
    plot.save("store/test-plot.jpg")
