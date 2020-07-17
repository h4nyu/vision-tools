import torch
from torch.utils.data import DataLoader
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.efficientdet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.models.box_merge import BoxMerge
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.data.object import TrainDataset
from object_detection.metrics import MeanPrecition

### config ###
confidence_threshold = 0.5
nms_threshold = 0.6
batch_size = 4
channels = 128

input_size = 512
object_count_range = (0, 20)
object_size_range = (32, 64)
iou_threshold = 0.55
### config ###


train_dataset = TrainDataset(
    (input_size, input_size),
    object_count_range=object_count_range,
    object_size_range=object_size_range,
    num_samples=1024,
)
test_dataset = TrainDataset(
    (input_size, input_size),
    object_count_range=object_count_range,
    object_size_range=object_size_range,
    num_samples=256,
)
backbone = EfficientNetBackbone(1, out_channels=channels, pretrained=True)
anchors = Anchors(size=2)
model = EfficientDet(
    num_classes=1, channels=channels, backbone=backbone, anchors=anchors
)
model_loader = ModelLoader(
    out_dir="/store/efficientdet", key="test_loss", best_watcher=BestWatcher(mode="max")
)
box_merge = BoxMerge(iou_threshold=iou_threshold)
criterion = Criterion()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
visualize = Visualize("/store/efficientdet", "test", limit=2)
get_score = MeanPrecition()
to_boxes = ToBoxes(
    nms_threshold=nms_threshold, confidence_threshold=confidence_threshold,
)
trainer = Trainer(
    model,
    DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True,
    ),
    DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=batch_size * 2, shuffle=True
    ),
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    criterion=criterion,
    get_score=get_score,
    device="cuda",
    to_boxes=to_boxes,
    box_merge=box_merge,
)
trainer(1000)
