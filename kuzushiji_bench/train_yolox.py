import os
import torch_optimizer as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, Inference
from vision_tools.backbone import CSPDarknet
from vision_tools.neck import CSPPAFPN
from vision_tools.assign import SimOTA
from vision_tools.meter import MeanReduceDict
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch
from kuzushiji_bench.data import (
    KuzushijiDataset,
    TrainTransform,
    Transform,
    read_train_rows,
    collate_fn,
    kfold,
)
from kuzushiji_bench.metric import Metric
from tqdm import tqdm


def main() -> None:
    seed_everything()
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    writer = SummaryWriter(os.path.join("runs", cfg.name))
    checkpoint = Checkpoint[YOLOX](
        root_path=os.path.join(cfg.root_dir, cfg.name),
        default_score=0.0,
        comparator=lambda a, b: a < b,
    )
    backbone = CSPDarknet(
        depth=cfg.depth,
        hidden_channels=cfg.hidden_channels,
    )
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg.feat_range[0]:cfg.feat_range[1]],
        strides=backbone.strides[cfg.feat_range[0]:cfg.feat_range[1]],
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
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    assign = SimOTA(**cfg.assign)
    criterion = Criterion(assign=assign, **cfg.criterion)
    optimizer = optim.AdaBound(model.parameters(), **cfg.optimizer)
    # train_step = TrainStep(
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     use_amp=cfg.use_amp,
    # )
    # validation_step = ValidationStep(
    #     criterion=criterion,
    # )
    annotations = read_train_rows(cfg.root_dir)
    train_rows, validation_rows = kfold(annotations, **cfg.fold)
    train_dataset = KuzushijiDataset(
        train_rows,
        transform=TrainTransform(cfg.image_size),
    )
    val_dataset = KuzushijiDataset(
        validation_rows,
        transform=Transform(cfg.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg.train_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg.val_loader,
    )
    to_device = ToDevice(cfg.device)

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=to_device,
        criterion=criterion,
        optimizer=optimizer,
        loader=train_loader,
        meter=MeanReduceDict(),
        writer=writer,
    )
    metric = Metric()

    eval_step = EvalStep[YOLOX, TrainBatch](
        to_device=to_device,
        loader=val_loader,
        metric=metric,
        writer=writer,
        inference=Inference(),
    )

    for epoch in range(cfg.num_epochs):
        train_step(model, epoch)
        eval_step(model, epoch)


if __name__ == "__main__":
    main()
