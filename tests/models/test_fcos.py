import torch
from object_detection.models.fcos import (
    centerness,
    FocsBoxes,
    Head,
    FCOS,
    FPN,
)
from object_detection.entities import ImageBatch
from object_detection.models.backbones.resnet import (
    ResNetBackbone,
)


def test_centerness() -> None:
    arg = FocsBoxes(torch.tensor([[10, 20, 5, 15]]))
    res = centerness(arg)
    assert (res - torch.tensor([0.6123])).abs()[0] < 1e-4


def test_head() -> None:
    channels = 32
    n_classes = 2
    fn = Head(depth=2, in_channels=channels, n_classes=n_classes)
    size = 128
    p3 = torch.rand(1, 32, size, size)
    p4 = torch.rand(1, 32, size * 2, size * 2)
    logit_maps, center_maps, box_maps = fn([p3, p4])
    assert logit_maps[0].shape == (1, n_classes, size, size)
    assert logit_maps[1].shape == (
        1,
        n_classes,
        size * 2,
        size * 2,
    )

    assert center_maps[0].shape == (1, 1, size, size)
    assert center_maps[1].shape == (
        1,
        1,
        size * 2,
        size * 2,
    )

    assert box_maps[0].shape == (1, 4, size, size)
    assert box_maps[1].shape == (1, 4, size * 2, size * 2)


def test_fcos() -> None:
    channels = 32
    size = 512
    backbone = ResNetBackbone("resnet34", out_channels=channels)
    fpn = FPN(channels=channels, depth=1)
    head = Head(in_channels=channels, n_classes=1, depth=1)
    fn = FCOS(backbone=backbone, fpn=fpn, head=head)
    image_batch = ImageBatch(torch.rand(1, 3, size, size))

    logits, centers, boxes = fn(image_batch)
    assert len(logits) == 5
    assert len(centers) == 5
    assert len(boxes) == 5
