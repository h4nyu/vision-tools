import pytest, torch
from vision_tools.mkmaps import MkGaussianMaps, MkPointMaps
from vision_tools.anchors import EmptyAnchors
from vision_tools.centernet import (
    ToBoxes,
)


def test_mkmaps() -> None:
    h, w = 1000, 1000
    gt_boxes = torch.tensor(
        [
            [0.201, 0.402, 0.11, 0.11],
            [0.301, 0.402, 0.11, 0.11],
        ]
    )
    gt_labels = torch.tensor([1, 0])
    to_boxes = ToBoxes(threshold=0.1)
    mkmaps = MkGaussianMaps(sigma=20.0, num_classes=2)
    hm = mkmaps([gt_boxes], [gt_labels], (h, w), (h * 10, w * 10))
    assert hm.shape == (1, 2, h, w)
    mk_anchors = EmptyAnchors()
    anchormap = mk_anchors(hm)
    boxmaps = torch.tensor([0, 0, 0.1, 0.1]).view(1, 4, 1, 1).expand(1, 4, h, w)

    box_batch, conf_batch, label_batch = to_boxes((hm, boxmaps, anchormap))
    merged, _ = torch.max(hm[0], dim=0)


def test_mk_point_maps() -> None:
    h, w = 1000, 1000
    gt_points = torch.tensor(
        [
            [0.201, 0.402],
            [0.301, 0.402],
        ]
    )
    gt_labels = torch.tensor([1, 0])
    mkmaps = MkPointMaps(sigma=20.0, num_classes=2)
    hm = mkmaps([gt_points], [gt_labels], h, w)
    assert hm.shape == (1, 2, h, w)
