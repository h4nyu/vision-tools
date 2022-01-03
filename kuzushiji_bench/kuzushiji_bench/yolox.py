from typing import Any
from omegaconf import OmegaConf
from vision_tools.backbone import CSPDarknet
from vision_tools.neck import CSPPAFPN
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.assign import SimOTA


def get_model(cfg: Any) -> YOLOX:
    backbone = CSPDarknet(
        depth=cfg.depth,
        hidden_channels=cfg.hidden_channels,
    )
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg.feat_range[0] : cfg.feat_range[1]],
        strides=backbone.strides[cfg.feat_range[0] : cfg.feat_range[1]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg.hidden_channels,
        num_classes=cfg.num_classes,
        feat_range=cfg.feat_range,
        box_iou_threshold=cfg.box_iou_threshold,
        score_threshold=cfg.score_threshold,
    )
    return model


def get_criterion(cfg: Any) -> Criterion:
    assign = SimOTA(**cfg.assign)
    criterion = Criterion(assign=assign, **cfg.criterion)
    return criterion
