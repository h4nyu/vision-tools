from typing import Tuple, List
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

batch_size = 16
channels = 128
lr = 1e-3

input_size = (256, 256)
object_count_range = (5, 20)
object_size_range = (32, 64)
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("test_loss", "min")

anchor_ratios = [0.5, 1.0, 2]
anchor_scales = [1.0, 1.25, 1.5, 1.75]
anchor_size = 4.0

confidence_threshold = 0.5
iou_threshold = 0.5
## criterion
label_weight = 2.0
pos_threshold = 0.4
size_threshold = 0.4
label_thresholds = (0.3, 0.4)

# model
out_ids:List[PyramidIdx] = [4, 5, 6]
