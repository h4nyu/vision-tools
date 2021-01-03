import pytest, torch
from object_detection.utils import DetectionPlot
from object_detection.entities.box import YoloBoxes, Labels, BoxMaps
from object_detection.models.mkmaps import MkGaussianMaps
from object_detection.models.anchors import EmptyAnchors
from object_detection.models.centernet import (
    Heatmaps,
    ToBoxes,
)


def test_mkmaps() -> None:

    h, w = 40, 40
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
    mkmaps = MkGaussianMaps(sigma=2.0, num_classes=2)
    hm = mkmaps([gt_boxes], [gt_labels], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 2, h, w)
    mk_anchors = EmptyAnchors()
    anchormap = mk_anchors(hm)
    boxmaps = BoxMaps(
        torch.tensor([0, 0, 0.1, 0.1])
        .view(1, 4, 1, 1)
        .expand(1, 4, h, w)
    )

    box_batch, conf_batch, label_batch = to_boxes(
        (hm, boxmaps, anchormap)
    )
    for i in range(2):
        plot = DetectionPlot(w=w, h=h)
        plot.with_image((hm[0, i] + 1e-4).log())
        plot.with_yolo_boxes(gt_boxes, color="blue")
        plot.with_yolo_boxes(
            box_batch[0], labels=label_batch[0], color="red"
        )
        plot.save(f"store/test-mk-gaussian-map-{i}.png")