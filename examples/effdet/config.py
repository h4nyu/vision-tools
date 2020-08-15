from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.7
batch_size = 16

# model
channels = 64
depth = 2
lr = 1e-4

input_size = (256, 256)
object_count_range = (5, 20)
object_size_range = (32, 64)
iou_threshold = 0.7
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [2 / 3, 3 / 2]
anchor_size = 3
