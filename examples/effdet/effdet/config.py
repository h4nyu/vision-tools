from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.5
batch_size = 16
channels = 128
lr = 1e-4

input_size = (256, 256)
object_count_range = (1, 20)
object_size_range = (32, 64)
iou_threshold = 0.6
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")

anchor_ratios = [3/4, 1, 4/3]
anchor_scales = [2.0, 4.0]
