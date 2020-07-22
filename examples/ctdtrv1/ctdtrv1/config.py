from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.models.centernetv1 import MkCrossMaps
from object_detection.model_loader import WatchMode

## heatmap
sigma = 0.2

lr = 1e-3
batch_size = 13
out_idx: PyramidIdx = 3
channels = 128
input_size = (256, 256)
metric: Tuple[str, WatchMode] = ("test_loss", "min")

heatmap_weight = 1.0
box_weight = 50.0
object_count_range = (5, 20)
object_size_range = (32, 64)
out_dir = "/store/centernetv1"
use_peak = True
box_depth = 2
iou_threshold = 0.4
to_boxes_threshold = 0.4
anchor_size = 1

mkmaps = MkCrossMaps()
