from typing import Tuple, List
from object_detection.entities import YoloBoxes, Confidences, ImageId
from object_detection.data.object import PredictionDataset
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.box_merge import BoxMerge
from object_detection.model_loader import ModelLoader, BestWatcher
from torch.utils.data import DataLoader
from object_detection.models.efficientdet import (
    prediction_collate_fn,
    EfficientDet,
    Predictor,
    ToBoxes,
    Anchors,
)
from . import config


def predict() -> Tuple[List[YoloBoxes], List[Confidences], List[ImageId]]:
    dataset = PredictionDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=1024,
    )
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=prediction_collate_fn,
        batch_size=config.batch_size,
        shuffle=False,
    )

    backbone = EfficientNetBackbone(1, out_channels=config.channels, pretrained=True)
    anchors = Anchors(size=config.anchor_size, ratios=config.anchor_ratios)
    model = EfficientDet(
        num_classes=1, channels=config.channels, backbone=backbone, anchors=anchors
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    box_merge = BoxMerge(iou_threshold=config.iou_threshold)

    to_boxes = ToBoxes(confidence_threshold=config.confidence_threshold,)
    predictor = Predictor(
        model=model,
        loader=data_loader,
        model_loader=model_loader,
        device="cuda",
        box_merge=box_merge,
        to_boxes=to_boxes,
    )
    return predictor()
