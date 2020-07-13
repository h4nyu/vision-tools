import torch
from torch.utils.data import DataLoader
from object_detection.models.centernetv1 import (
    collate_fn,
    CenterNetV1,
    Visualize,
    Trainer,
    Criterion,
    ToBoxes,
    Anchors,
    BoxMerge,
)
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.model_loader import ModelLoader
from object_detection.data.object import ObjectDataset
from object_detection.metrics import MeanPrecition
from object_detection.meters import BestWatcher
from . import config as cfg


train_dataset = ObjectDataset(
    cfg.input_size,
    object_count_range=cfg.object_count_range,
    object_size_range=cfg.object_size_range,
    num_samples=1024,
)
test_dataset = ObjectDataset(
    cfg.input_size,
    object_count_range=cfg.object_count_range,
    object_size_range=cfg.object_size_range,
    num_samples=256,
)
backbone = ResNetBackbone("resnet50", out_channels=cfg.channels)
model = CenterNetV1(
    channels=cfg.channels,
    backbone=backbone,
    out_idx=cfg.out_idx,
    box_depth=cfg.box_depth,
    anchors=Anchors(size=cfg.anchor_size),
)
model_loader = ModelLoader(cfg.out_dir)
criterion = Criterion(
    box_weight=cfg.box_weight, heatmap_weight=cfg.heatmap_weight, sigma=cfg.sigma,
)
train_loader = DataLoader(
    train_dataset, collate_fn=collate_fn, batch_size=cfg.batch_size, shuffle=True
)
test_loader = DataLoader(
    test_dataset, collate_fn=collate_fn, batch_size=cfg.batch_size * 2, shuffle=True
)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
visualize = Visualize(cfg.out_dir, "test", limit=2)
best_watcher = BestWatcher(mode="max")
to_boxes = ToBoxes(threshold=cfg.to_boxes_threshold)
box_merge = BoxMerge(iou_threshold=cfg.iou_threshold)
get_score = MeanPrecition()
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    criterion=criterion,
    best_watcher=best_watcher,
    device="cuda",
    get_score=get_score,
    box_merge=box_merge,
    to_boxes=to_boxes,
)
