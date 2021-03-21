from typing import *
from vnet.centernet import ToPoints, HMLoss
from vnet.mkmaps import MkPointMaps
from vnet.backbones.resnet import (
    ResNetBackbone,
)
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet.model_loader import WatchMode
from .model import Net

num_classes = 2
## heatmap
sigma = 1.0
use_peak = True
confidence_threshold = 0.3

lr = 1e-3
batch_size = 8
out_idx = 4
channels = 128
input_size = (512, 512)
metric: Tuple[str, WatchMode] = ("score", "max")

heatmap_weight = 1.0
box_weight = 5.0
object_count_range = (5, 10)
object_size_range = (32, 64)
out_dir = "/store/centernet_kp"
box_depth = 1
cls_depth = 2


backbone = ResNetBackbone("resnet50", out_channels=channels)
net = Net(
    backbone=backbone,
    num_classes=num_classes,
    channels=channels,
    out_idx=out_idx,
    cls_depth=cls_depth,
)
to_points = ToPoints(threshold=confidence_threshold)
mkmaps = MkPointMaps(
    num_classes=num_classes,
    sigma=sigma,
)
hmloss = HMLoss()

model_loader = ModelLoader(
    out_dir=out_dir,
    key="score",
    best_watcher=BestWatcher(mode="max"),
)
