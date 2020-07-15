from object_detection.models.centernetv1 import Predictor, prediction_collate_fn
from object_detection.data.object import PredictionDataset
from object_detection.models.centernetv1 import (
    BoxMerge
)
from torch.utils.data import DataLoader
from . import config as cfg
from . import train

dataset = PredictionDataset(
    cfg.input_size,
    object_count_range=cfg.object_count_range,
    object_size_range=cfg.object_size_range,
    num_samples=1024,
)

data_loader = DataLoader(
    dataset=dataset,
    collate_fn=prediction_collate_fn,
    batch_size=cfg.batch_size,
    shuffle=True,
)
box_merge = BoxMerge(iou_threshold=cfg.iou_threshold)

predictor = Predictor(
    model=train.model,
    loader=data_loader,
    model_loader=train.model_loader,
    device="cuda",
    box_merge=box_merge,
    to_boxes=train.to_boxes,
)
