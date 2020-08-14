from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.models.centernetv1 import GaussianMapMode
from object_detection.model_loader import WatchMode

## heatmap
sigma = 1.0
use_peak = True
mode: GaussianMapMode = "aspect"
to_boxes_threshold = 0.1

lr = 1e-3
batch_size = 16
out_idx: PyramidIdx = 4
channels = 256
input_size = (256, 256)
metric: Tuple[str, WatchMode] = ("score", "max")

heatmap_weight = 1.0
box_weight = 50.0
object_count_range = (5, 20)
object_size_range = (32, 64)
out_dir = "/store/centernet"
box_depth = 2
iou_threshold = 0.4
anchor_size = 1
