import os
import torch_optimizer as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, TrainStep
from vision_tools.interface import TrainBatch, TrainSample
from vision_tools.backbone import CSPDarknet
from vision_tools.neck import CSPPAFPN
from vision_tools.assign import SimOTA
from vision_tools.meter import MeanReduceDict
from kuzushiji_bench.data import KuzushijiDataset, TrainTransform, read_train_rows


def collate_fn(
    batch: list[TrainSample],
) -> TrainBatch:
    images: list[Tensor] = []
    box_batch: list[Tensor] = []
    label_batch: list[Tensor] = []
    for row in batch:
        images.append(row["image"])
        box_batch.append(row["boxes"])
        label_batch.append(row["labels"])
    return TrainBatch(
        image_batch=torch.stack(images),
        box_batch=box_batch,
        label_batch=label_batch,
    )


def main() -> None:
    seed_everything()
    # logger = getLogger(cfg.name)
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    writer = SummaryWriter(os.path.join("runs", cfg.name))
    checkpoint = Checkpoint[YOLOX](
        root_path=os.path.join(cfg.root_dir, cfg.name),
        default_score=0.0,
    )
    backbone = CSPDarknet()
    neck = CSPPAFPN(
        in_channels=backbone.channels,
        strides=backbone.strides,
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=64,
        num_classes=cfg.num_classes,
        box_feat_range=cfg.box_feat_range,
        box_iou_threshold=cfg.box_iou_threshold,
        score_threshold=cfg.score_threshold,
    )
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    assign = SimOTA(**cfg.assign)
    criterion = Criterion(model=model, assign=assign, **cfg.criterion)
    optimizer = optim.AdaBound(model.parameters(), **cfg.optimizer)
    train_step = TrainStep(
        optimizer=optimizer,
        criterion=criterion,
        use_amp=cfg.use_amp,
    )
    # validation_step = ValidationStep(
    #     criterion=criterion,
    # )
    train_dataset = KuzushijiDataset(
        read_train_rows(cfg.root_dir),
        transform=TrainTransform(cfg.image_size),
    )
    # val_dataset = CellTrainDataset(
    #     **cfg.dataset,
    #     transform=Tranform(
    #         size=cfg.patch_size,
    #     ),
    # )
    # train_indecies, validation_indecies = get_fold_indices(train_dataset, **cfg.fold)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg.train_loader,
    )
    # val_loader = DataLoader(
    #     Subset(val_dataset, validation_indecies),
    #     collate_fn=collate_fn,
    #     **cfg.validation_loader,
    # )
    to_device = ToDevice(cfg.device)

    for _ in range(cfg.num_epochs):
        train_reduer = MeanReduceDict()
        for batch in train_loader:
            batch = to_device(**batch)
            train_log = train_step(batch)
            train_reduer.accumulate(train_log)
        writer.add_scalars("loss", train_reduer.value)
    #     mask_ap = MaskAP(**cfg.mask_ap)
    #     for batch in val_loader:
    #         batch = to_device(*batch)
    #         validation_log = validation_step(batch, on_end=mask_ap.accumulate_batch)
    #     if score < mask_ap.value:
    #         score = checkpoint.save(model, mask_ap.value)
    #         logger.info(f"save checkpoint")
    #     logger.info(f"epoch eval {score=} {mask_ap.value=}")


if __name__ == "__main__":
    main()
