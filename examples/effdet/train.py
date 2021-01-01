import torch
from torch.utils.data import DataLoader
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.models.effidet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from examples.data import TrainDataset
from object_detection.metrics import MeanPrecition
from examples.effdet import config


def train(epochs: int) -> None:
    train_dataset = TrainDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=1024,
    )
    test_dataset = TrainDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=256,
    )
    backbone = EfficientNetBackbone(
        config.backbone_id,
        out_channels=config.channels,
        pretrained=True,
    )
    anchors = Anchors(
        size=config.anchor_size,
        ratios=config.anchor_ratios,
        scales=config.anchor_scales,
    )
    model = EfficientDet(
        num_classes=1,
        out_ids=config.out_ids,
        channels=config.channels,
        backbone=backbone,
        anchors=anchors,
        box_depth=config.box_depth,
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion(
        topk=config.topk,
        box_weight=config.box_weight,
        cls_weight=config.cls_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    visualize = Visualize("/store/efficientdet", "test", limit=2)
    get_score = MeanPrecition()
    to_boxes = ToBoxes(
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
    )
    trainer = Trainer(
        model,
        DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=config.batch_size,
            shuffle=True,
        ),
        DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_size=config.batch_size * 2,
            shuffle=True,
        ),
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        criterion=criterion,
        get_score=get_score,
        device="cuda",
        to_boxes=to_boxes,
    )
    trainer(epochs)


if __name__ == "__main__":
    train(10000)
