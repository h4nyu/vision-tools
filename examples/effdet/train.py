import torch
from typing import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from vnet.backbones.effnet import (
    EfficientNetBackbone,
)
from vnet import (
    Image,
    ImageBatch,
    Boxes,
    Labels,
    Boxes,
)
from vnet.metrics import MeanAveragePrecision
from tqdm import tqdm
from vnet.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet.meters import MeanMeter
from examples.data import BoxDataset
import torch_optimizer as optim
from examples.effdet import config
from logging import (
    getLogger,
)

logger = getLogger(__name__)


def collate_fn(
    batch: list[tuple[str, Image, Boxes, Labels]],
) -> tuple[ImageBatch, list[Boxes], list[Labels], list[str]]:
    images: list[Any] = []
    id_batch: list[str] = []
    box_batch: list[Boxes] = []
    label_batch: list[Labels] = []
    for id, img, boxes, labels in batch:
        c, h, w = img.shape
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return (
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
        id_batch,
    )


def train(epochs: int) -> None:
    device = torch.device("cuda")
    train_dataset = BoxDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=1024,
    )
    test_dataset = BoxDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=256,
    )
    backbone = EfficientNetBackbone(
        config.backbone_id,
        out_channels=config.channels,
        pretrained=True,
    )
    anchors = Anchors(
        size=config.anchor_size,
        ratios=config.anchor_ratios,
        scales=config.anchor_scales,
    )
    model = EfficientDet(
        num_classes=config.num_classes,
        out_ids=config.out_ids,
        channels=config.channels,
        backbone=backbone,
        anchors=anchors,
        box_depth=config.box_depth,
    ).to(device)
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion(
        topk=config.topk,
        box_weight=config.box_weight,
        cls_weight=config.cls_weight,
    )
    optimizer = optim.RAdam(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0,
    )
    visualize = Visualize(
        "/store/efficientdet",
        "test",
        limit=config.batch_size,
        box_limit=config.vis_box_limit,
    )
    to_boxes = ToBoxes(
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 2,
        shuffle=True,
    )
    scaler = GradScaler()
    logs: dict[str, float] = {}

    def train_step() -> None:
        model.train()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        for (
            image_batch,
            gt_box_batch,
            gt_label_batch,
            _,
        ) in tqdm(train_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                netout = model(image_batch)
                loss, box_loss, label_loss = criterion(
                    image_batch,
                    netout,
                    gt_box_batch,
                    gt_label_batch,
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
        metrics = MeanAveragePrecision(
            iou_threshold=0.3, num_classes=config.num_classes
        )
        for image_batch, gt_box_batch, gt_label_batch, _ in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            netout = model(image_batch)
            loss, box_loss, label_loss = criterion(
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
                    boxes=boxes,
                    confidences=confidences,
                    labels=labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )

        score, scores = metrics()
        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"] = score
        for k, v in scores.items():
            logs[f"score-{k}"] = v

        visualize(
            image_batch,
            (box_batch, confidence_batch, label_batch),
            (gt_box_batch, gt_label_batch),
        )
        score, scores = metrics()
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
    train(10000)
