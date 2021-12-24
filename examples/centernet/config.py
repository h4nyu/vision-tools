from typing import *
from vision_tools.mkmaps import (
    GaussianMapMode,
)
from vision_tools.model_loader import WatchMode

num_classes = 2
## heatmap
sigma = 5.0
use_peak = True
mode: GaussianMapMode = "aspect"
to_boxes_threshold = 0.3

lr = 1e-3
batch_size = 8
out_idx = 4
channels = 128
input_size = (256, 256)
metric: tuple[str, WatchMode] = ("score", "max")

heatmap_weight = 1.0
box_weight = 5.0
object_count_range = (5, 10)
object_size_range = (32, 64)
out_dir = "/store/centernet"
box_depth = 1
cls_depth = 2
