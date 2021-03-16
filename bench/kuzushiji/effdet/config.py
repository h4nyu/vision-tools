import os, vnet
from bench.kuzushiji.config import Config as BaseConfig
from typing import Any
import torch_optimizer as optim
import dataclasses
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet.backbones.effnet import (
    EfficientNetBackbone,
)
from vnet.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)


@dataclasses.dataclass
class Config(BaseConfig):
    batch_size = 6
    use_amp = True
    out_ids = [5, 6]
    lr = 1e-4
    channels = 128
    box_depth = 1
    backbone_id = 4
    confidence_threshold = 0.3
    iou_threshold = 0.5
    anchor_size = 2
    anchor_ratios = [1.0]
    anchor_scales = [1.0]
    box_offset = -2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.out_dir = os.path.join(self.root_dir, "effidet")
        self.submission_path = os.path.join(self.out_dir, "submission.csv")

        backbone = EfficientNetBackbone(
            self.backbone_id,
            out_channels=self.channels,
            pretrained=True,
        )
        anchors = Anchors(
            size=self.anchor_size,
            ratios=self.anchor_ratios,
            scales=self.anchor_scales,
        )
        self.model = EfficientDet(
            num_classes=self.num_classes,
            out_ids=self.out_ids,
            channels=self.channels,
            backbone=backbone,
            anchors=anchors,
            box_depth=self.box_depth,
        ).to(self.device)
        self.to_points = vnet.to_center_points

        self.optimizer = optim.RAdam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0,
        )

        self.model_loader = ModelLoader(
            out_dir=self.out_dir,
            key="score",
            best_watcher=BestWatcher(
                # mode="min"
                mode="max",
            ),
        )

        self.to_boxes = ToBoxes(
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
        self.box_padding = lambda b: vnet.box_padding(b, -6)
        self.criterion = Criterion(
            topk=anchors.num_anchors * len(self.out_ids) * 10,
            box_weight=0.5,
            cls_weight=1,
        )

    def load(self, path: str) -> None:
        ...
