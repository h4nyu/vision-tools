import os
from datetime import datetime
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from vision_tools.assign import SimOTA
from vision_tools.backbone import CSPDarknet, EfficientNet
from vision_tools.neck import CSPPAFPN
from vision_tools.utils import Checkpoint
from vision_tools.yolox import YOLOX, Criterion


def get_model_name(cfg: Any) -> str:
    return f"{cfg['name']}-{cfg['feat_range'][0]}-{cfg['feat_range'][1]}-{cfg['hidden_channels']}-{cfg['backbone']}"


def get_writer(cfg: Any) -> SummaryWriter:
    model_name = get_model_name(cfg)
    return SummaryWriter(
        f"runs/{model_name}-lr_{cfg['lr']}-box_w_{cfg['criterion']['box_weight']}-radius_{cfg['assign']['radius']}"
    )


def get_model(cfg: Any) -> YOLOX:
    # backbone = CSPDarknet(
    #     depth=cfg.depth,
    #     hidden_channels=cfg.hidden_channels,
    # )
    backbone = EfficientNet(name=cfg["backbone_name"])
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg["feat_range"][0] : cfg["feat_range"][1]],
        strides=backbone.strides[cfg["feat_range"][0] : cfg["feat_range"][1]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg["hidden_channels"],
        num_classes=cfg["num_classes"],
        feat_range=cfg["feat_range"],
        box_iou_threshold=cfg["box_iou_threshold"],
        score_threshold=cfg["score_threshold"],
    )
    return model


def get_criterion(cfg: Any) -> Criterion:
    assign = SimOTA(**cfg.assign)
    criterion = Criterion(assign=assign, **cfg.criterion)
    return criterion


def get_checkpoint(cfg: Any) -> Checkpoint:
    return Checkpoint[YOLOX](
        root_path=os.path.join(cfg.root_dir, get_model_name(cfg)),
        default_score=0.0,
    )
