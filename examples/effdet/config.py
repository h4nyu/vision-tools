from typing import *
from object_detection.model_loader import WatchMode

# train
confidence_threshold = 0.4
iou_threshold = 0.60
batch_size = 8
lr = 1e-3
use_amp = True

# model
num_classes = 2
backbone_id = 1
channels = 128
box_depth = 1
out_ids: List[int] = [6, 7]


input_size = (256, 256)
object_count_range = (0, 2)
object_size_range = (32, 64)
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

# criterion
topk = 9
box_weight = 20
cls_weight = 1

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.33, 1.66]
anchor_size = 2
