import torch
from typing import *
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from vnet.meters import MeanMeter
from vnet.centernet import (
    CenterNet,
    Visualize,
    Criterion,
    ToBoxes,
)
from vnet import (
    yolo_to_pascal,
)
import torch_optimizer as optim
from vnet.mkmaps import (
    MkGaussianMaps,
    MkCenterBoxMaps,
)
from vnet.backbones.resnet import (
    ResNetBackbone,
)
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet import (
    Image,
    Boxes,
    Labels,
    ImageBatch,
    YoloBoxes,
    pascal_to_yolo,
)
from examples.data import BoxDataset
from vnet.metrics import MeanAveragePrecision
from examples.centernet import config as cfg
from logging import (
    getLogger,
)

logger = getLogger(__name__)


def collate_fn(
    batch: list[tuple[str, Image, Boxes, Labels]],
) -> tuple[list[str], ImageBatch, list[YoloBoxes], list[Labels]]:
    images: list[Any] = []
    id_batch: list[str] = []
    box_batch: list[YoloBoxes] = []
    label_batch: list[Labels] = []

    for id, img, boxes, labels in batch:
        images.append(img)
        _, h, w = img.shape
        box_batch.append(pascal_to_yolo(boxes, (w, h)))
        id_batch.append(id)
        label_batch.append(labels)
    return (
        id_batch,
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
    )


def train(epochs: int) -> None:
    device = "cuda"
    use_amp = True
    train_dataset = BoxDataset(
        cfg.input_size,
        object_count_range=cfg.object_count_range,
        object_size_range=cfg.object_size_range,
        num_samples=1024,
    )
    test_dataset = BoxDataset(
        cfg.input_size,
        object_count_range=cfg.object_count_range,
        object_size_range=cfg.object_size_range,
        num_samples=256,
    )
    backbone = ResNetBackbone("resnet50", out_channels=cfg.channels)
    model = CenterNet(
        num_classes=2,
        channels=cfg.channels,
        backbone=backbone,
        out_idx=cfg.out_idx,
        box_depth=cfg.box_depth,
        cls_depth=cfg.cls_depth,
    ).to(device)
    criterion = Criterion(
        box_weight=cfg.box_weight,
        heatmap_weight=cfg.heatmap_weight,
        mk_hmmaps=MkGaussianMaps(num_classes=cfg.num_classes, sigma=cfg.sigma),
        mk_boxmaps=MkCenterBoxMaps(),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size * 2,
        shuffle=True,
    )
    optimizer = optim.RAdam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    visualize = Visualize(cfg.out_dir, "test", limit=cfg.batch_size)
    metrics = MeanAveragePrecision(iou_threshold=0.3, num_classes=cfg.num_classes)
    logs: dict[str, float] = {}
    model_loader = ModelLoader(
        out_dir=cfg.out_dir,
        key=cfg.metric[0],
        best_watcher=BestWatcher(mode=cfg.metric[1]),
    )
    to_boxes = ToBoxes(threshold=cfg.to_boxes_threshold)
    scaler = GradScaler()

    def train_step() -> None:
        model.train()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(train_loader):
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            image_batch = image_batch.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                netout = model(image_batch)
                loss, label_loss, box_loss, _ = criterion(
                    image_batch, netout, gt_box_batch, gt_label_batch
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())

        logs["train_loss"] = loss_meter.get_value()
        logs["train_box"] = box_loss_meter.get_value()
        logs["train_label"] = label_loss_meter.get_value()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        metrics = MeanAveragePrecision(iou_threshold=0.3, num_classes=cfg.num_classes)
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            _, _, h, w = image_batch.shape
            netout, _ = model(image_batch)
            loss, label_loss, box_loss, gt_hms = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = to_boxes(netout)

            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())
            for boxes, gt_boxes, labels, gt_labels, confidences in zip(
                box_batch, gt_box_batch, label_batch, gt_label_batch, confidence_batch
            ):
                metrics.add(
                    boxes=yolo_to_pascal(boxes, (w, h)),
                    confidences=confidences,
                    labels=labels,
                    gt_boxes=yolo_to_pascal(gt_boxes, (w, h)),
                    gt_labels=gt_labels,
                )

        visualize(
            netout,
            box_batch,
            confidence_batch,
            label_batch,
            gt_box_batch,
            gt_label_batch,
            image_batch,
            gt_hms,
        )
        score, scores = metrics()
        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"] = score
        for k, v in scores.items():
            logs[f"score-{k}"] = v
        model_loader.save_if_needed(
            model,
            score,
        )

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()
        eval_step()
        log()


if __name__ == "__main__":
    train(1000)
