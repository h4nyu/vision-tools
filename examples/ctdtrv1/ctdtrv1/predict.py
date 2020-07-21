from object_detection.models.centernetv1 import (
    Predictor,
    prediction_collate_fn,
    CenterNetV1,
    ToBoxes,
    Anchors,
)
from typing import Tuple, List
from object_detection.entities import YoloBoxes, Confidences, ImageId
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.data.object import PredictionDataset
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.box_merge import BoxMerge
from torch.utils.data import DataLoader
from . import config as cfg


def predict() -> Tuple[List[YoloBoxes], List[Confidences], List[ImageId]]:
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
        shuffle=False,
    )

    backbone = EfficientNetBackbone(1, out_channels=cfg.channels, pretrained=True)
    model = CenterNetV1(
        channels=cfg.channels,
        backbone=backbone,
        out_idx=cfg.out_idx,
        box_depth=cfg.box_depth,
        anchors=Anchors(size=cfg.anchor_size),
    )
    model_loader = ModelLoader(
        out_dir=cfg.out_dir,
        key=cfg.metric[0],
        best_watcher=BestWatcher(mode=cfg.metric[1]),
    )
    box_merge = BoxMerge(iou_threshold=cfg.iou_threshold)

    to_boxes = ToBoxes(threshold=cfg.to_boxes_threshold,)
    predictor = Predictor(
        model=model,
        loader=data_loader,
        model_loader=model_loader,
        device="cuda",
        box_merge=box_merge,
        to_boxes=to_boxes,
    )
    return predictor()
