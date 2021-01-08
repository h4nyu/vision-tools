from typing import Tuple
from object_detection.models.mkmaps import (
    GaussianMapMode,
)
from object_detection.model_loader import WatchMode

num_classes = 2
## heatmap
sigma = 10.0
use_peak = True
mode: GaussianMapMode = "aspect"
to_boxes_threshold = 0.3

lr = 1e-3
batch_size = 24
out_idx = 4
channels = 64
input_size = (512, 512)
metric: Tuple[str, WatchMode] = ("score", "max")

heatmap_weight = 1.0
box_weight = 5.0
object_count_range = (5, 20)
object_size_range = (32, 128)
out_dir = "/store/centernet"
box_depth = 1
cls_depth = 2
