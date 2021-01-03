from typing import *
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.3
iou_threshold = 0.66
batch_size = 8

# model
backbone_id = 1
channels = 64
box_depth = 2
lr = 1e-3
out_ids: List[int] = [4, 5, 6]


input_size = (256, 256)
object_count_range = (5, 20)
object_size_range = (32, 64)
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 39
box_weight = 10
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.25, 1.5, 1.75]
anchor_size = 2
