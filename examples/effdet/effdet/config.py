from typing import Tuple, List
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

batch_size = 20
channels = 128
lr = 1e-4

input_size = (256, 256)
object_count_range = (1, 20)
object_size_range = (32, 64)
out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("test_loss", "min")

anchor_ratios = [1.0]
anchor_scales = [2.0]

confidence_threshold = 0.3
iou_threshold = 0.5
## criterion
label_weight = 10.0
pos_threshold = 0.4
size_threshold = 0.4
label_thresholds = (0.0, 0.3)

# model
out_ids:List[PyramidIdx] = [5, 6]
