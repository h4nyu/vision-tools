import torch, time, gc, pytest
from typing import *
from object_detection.entities.image import ImageBatch
from object_detection.entities.box import PascalBoxes, Labels
from object_detection.models.effidet import (
    RegressionModel,
    ClassificationModel,
    EfficientDet,
    Criterion,
    BoxDiff,
)
from object_detection.models.anchors import Anchors
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)


def test_regression_model() -> None:
    c, h, w = 4, 10, 10
    num_anchors = 9
    images = torch.ones((1, c, h, w))
    fn = RegressionModel(
        in_channels=c, num_anchors=num_anchors, out_size=c
    )
    res = fn(images)
    assert res.shape == (1, h * w * num_anchors, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(in_channels=100, num_classes=2)
    res = fn(images)
    assert res.shape == (1, 900, 2)


def test_effdet() -> None:
    images = ImageBatch(torch.ones((1, 3, 512, 512)))
    annotations = torch.ones((1, 10, 5))
    channels = 32
    backbone = EfficientNetBackbone(
        1, out_channels=channels, pretrained=True
    )
    fn = EfficientDet(
        num_classes=2,
        backbone=backbone,
        channels=32,
    )
    anchors, boxes, labels = fn(images)
    for x, y in zip(labels, boxes):
        assert x.shape[:2] == y.shape[:2]


def make_model(in_size: int, out_size: int, num_layers: int) -> Any:
    layers: List[Any] = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


start_time = 0.0


def start_timer() -> None:
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg: str = "") -> None:
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print(
        "Total execution time = {:.3f} sec".format(
            end_time - start_time
        )
    )
    print(
        "Max memory used by tensors = {} bytes".format(
            torch.cuda.max_memory_allocated()
        )
    )


def test_amp() -> None:
    loss_fn = torch.nn.MSELoss().cuda()
    batch_size = 512  # Try, for example, 128, 256, 513.
    in_size = 4096
    out_size = 4096
    num_layers = 10
    num_batches = 50
    epochs = 3
    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=0.001)
    data = [
        torch.randn(batch_size, in_size, device="cuda")
        for _ in range(num_batches)
    ]
    targets = [
        torch.randn(batch_size, out_size, device="cuda")
        for _ in range(num_batches)
    ]

    start_timer()
    for epoch in range(epochs):
        for input, target in zip(data, targets):
            output = net(input)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            opt.zero_grad()  # set_to_none=True here can modestly improve performance
    end_timer_and_print("Default precision:")

    scaler = torch.cuda.amp.GradScaler()
    start_timer()
    for epoch in range(epochs):
        for input, target in zip(data, targets):
            with torch.cuda.amp.autocast(enabled=True):
                output = net(input)
                assert output.dtype is torch.float16
                loss = loss_fn(output, target)
                assert loss.dtype is torch.float32
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()  # set_to_none=True here can modestly improve performance
    end_timer_and_print("Mixed precision:")
