from object_detection.entities import PyramidIdx

sigma = 5.0
lr = 1e-4
batch_size = 16
out_idx: PyramidIdx = 4
channels = 256
input_size = (256, 256)
heatmap_weight = 1.0
box_weight = 50.0
object_count_range = (1, 20)
object_size_range = (32, 64)
out_dir = "/store/centernetv1"
box_depth = 2
iou_threshold = 0.4
to_boxes_threshold = 0.4
anchor_size = 1
