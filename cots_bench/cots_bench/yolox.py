import os
from typing import Any, Dict
import torch
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, Inference
from vision_tools.backbone import CSPDarknet, EfficientNet
from vision_tools.neck import CSPPAFPN
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.assign import SimOTA
from vision_tools.utils import Checkpoint, load_config
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from vision_tools.meter import MeanReduceDict
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    read_train_rows,
    collate_fn,
    kfold,
)
from cots_bench.metric import Metric



def get_model_name(cfg: Dict[str, Any]) -> str:
    return f"{cfg['name']}-{cfg['feat_range'][0]}-{cfg['feat_range'][1]}-{cfg['hidden_channels']}-{cfg['backbone_name']}"


def get_writer(cfg: Dict[str, Any]) -> SummaryWriter:
    model_name = get_model_name(cfg)
    return SummaryWriter(
        f"/app/runs/{model_name}-lr_{cfg['lr']}-box_w_{cfg['criterion']['box_weight']}-radius_{cfg['assign']['radius']}"
    )


def get_model(cfg: Dict[str, Any]) -> YOLOX:
    backbone = EfficientNet(name=cfg['backbone_name'])
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg['feat_range'][0] : cfg['feat_range'][1]],
        strides=backbone.strides[cfg['feat_range'][0] : cfg['feat_range'][1]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg['hidden_channels'],
        num_classes=cfg['num_classes'],
        feat_range=cfg['feat_range'],
        box_iou_threshold=cfg['box_iou_threshold'],
        score_threshold=cfg['score_threshold'],
    )
    return model


def get_criterion(cfg: Dict[str, Any]) -> Criterion:
    assign = SimOTA(**cfg['assign'])
    criterion = Criterion(assign=assign, **cfg['criterion'])
    return criterion


def get_checkpoint(cfg: Dict[str, Any]) -> Checkpoint:
    return Checkpoint[YOLOX](
        root_path=os.path.join(cfg['root_dir'], get_model_name(cfg)),
        default_score=0.0,
    )


def train() -> None:
    seed_everything()
    cfg = load_config(os.path.join(os.path.dirname(__file__), "../config/yolox.yaml"))
    checkpoint = get_checkpoint(cfg)
    writer = get_writer(cfg)
    model = get_model(cfg)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    checkpoint.load_if_exists(model=model, optimizer=optimizer, device=cfg['device'])

    annotations = read_train_rows(cfg['root_dir'])
    train_rows, validation_rows = kfold(annotations, cfg['n_splits'], cfg['fold'])
    train_dataset = COTSDataset(
        train_rows,
        transform=TrainTransform(cfg['image_size']),
        image_dir=cfg['image_dir'],
    )
    val_dataset = COTSDataset(
        validation_rows,
        transform=Transform(cfg['image_size']),
        image_dir=cfg['image_dir'],
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg['train_loader'],
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg['val_loader'],
    )
    to_device = ToDevice(cfg['device'])

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=to_device,
        criterion=criterion,
        optimizer=optimizer,
        loader=train_loader,
        meter=MeanReduceDict(),
        writer=writer,
        checkpoint=checkpoint,
        use_amp=cfg['use_amp'],
    )
    metric = Metric()

    eval_step = EvalStep[YOLOX, TrainBatch](
        to_device=to_device,
        loader=val_loader,
        metric=metric,
        writer=writer,
        inference=Inference(),
        checkpoint=checkpoint,
    )

    for epoch in range(cfg['num_epochs']):
        train_step(model, epoch)
        eval_step(model, epoch)
