import os
from bench.kuzushiji.config import *
import torch_optimizer as optim
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet.backbones.effnet import (
    EfficientNetBackbone,
)
from vnet.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)

batch_size = 14
use_amp = True

out_ids = [5, 6]
lr = 5e-4
channels = 128
box_depth = 1
backbone_id = 4
confidence_threshold = 0.3
iou_threshold = 0.5
anchor_size = 1
anchor_ratios = [1.0]
anchor_scales = [1.0]

out_dir = os.path.join(root_dir, "effidet")

backbone = EfficientNetBackbone(
    backbone_id,
    out_channels=channels,
    pretrained=True,
)
anchors = Anchors(
    size=anchor_size,
    ratios=anchor_ratios,
    scales=anchor_scales,
)
model = EfficientDet(
    num_classes=num_classes,
    out_ids=out_ids,
    channels=channels,
    backbone=backbone,
    anchors=anchors,
    box_depth=box_depth,
).to(device)

optimizer = optim.RAdam(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-16,
    weight_decay=0,
)

model_loader = ModelLoader(
    out_dir=out_dir,
    key="score",
    best_watcher=BestWatcher(mode="max"),
)


to_boxes = ToBoxes(
    confidence_threshold=confidence_threshold,
    iou_threshold=iou_threshold,
)

criterion = Criterion(
    topk=anchors.num_anchors * len(out_ids) * 10,
    box_weight=1,
    cls_weight=1,
)
