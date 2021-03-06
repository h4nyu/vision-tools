import pytest, torch
from vnet.utils import DetectionPlot
from vnet import YoloBoxes, Labels, BoxMaps, yolo_to_pascal, Points
from vnet.mkmaps import MkGaussianMaps, MkPointMaps
from vnet.anchors import EmptyAnchors
from vnet.centernet import (
    Heatmaps,
    ToBoxes,
)


def test_mkmaps() -> None:
    h, w = 1000, 1000
    gt_boxes = YoloBoxes(
        torch.tensor(
            [
                [0.201, 0.402, 0.11, 0.11],
                [0.301, 0.402, 0.11, 0.11],
            ]
        )
    )
    gt_labels = Labels(torch.tensor([1, 0]))
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkGaussianMaps(sigma=20.0, num_classes=2)
    hm = mkmaps([gt_boxes], [gt_labels], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 2, h, w)
    mk_anchors = EmptyAnchors()
    anchormap = mk_anchors(hm)
    boxmaps = BoxMaps(
        torch.tensor([0, 0, 0.1, 0.1]).view(1, 4, 1, 1).expand(1, 4, h, w)
    )

    box_batch, conf_batch, label_batch = to_boxes((hm, boxmaps, anchormap))
    merged, _ = torch.max(hm[0], dim=0)
    plot = DetectionPlot(merged * 255)
    plot.draw_boxes(yolo_to_pascal(gt_boxes, (w, h)), color="blue")
    plot.draw_boxes(yolo_to_pascal(box_batch[0], (w, h)), color="red")
    plot.save(f"store/test-mk-gaussian-map.png")


def test_mk_point_maps() -> None:
    h, w = 1000, 1000
    gt_points = Points(
        torch.tensor(
            [
                [0.201, 0.402],
                [0.301, 0.402],
            ]
        )
    )
    gt_labels = Labels(torch.tensor([1, 0]))
    mkmaps = MkPointMaps(sigma=20.0, num_classes=2)
    hm = mkmaps([gt_points], [gt_labels], h, w)
    assert hm.shape == (1, 2, h, w)
    plot = DetectionPlot(hm[0, 0])
    plot.save(f"store/test_mk_point_maps.png")
