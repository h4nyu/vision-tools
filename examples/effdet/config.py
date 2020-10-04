from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.5
batch_size = 8

# model
channels = 256
depth = 2
lr = 1e-4

input_size = (256, 256)
object_count_range = (5, 20)
object_size_range = (32, 64)
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0]
anchor_size = 1
