import torch
from typing import *
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from object_detection.meters import MeanMeter
from object_detection.utils import DetectionPlot
import torch_optimizer as optim
from object_detection import (
    Image,
    Points,
    Labels,
    ImageBatch,
    Points,
    pascal_to_yolo,
    resize_points,
)
from examples.data import PointDataset
from object_detection.metrics import MeanAveragePrecision
from examples.centernet_kp import config
from logging import (
    getLogger,
)

logger = getLogger(__name__)


def collate_fn(
    batch: list[tuple[str, Image, Points, Labels]],
) -> tuple[list[str], ImageBatch, list[Points], list[Labels]]:
    images: list[Any] = []
    id_batch: list[str] = []
    point_batch: list[Points] = []
    label_batch: list[Labels] = []

    for id, img, points, labels in batch:
        images.append(img)
        _, h, w = img.shape
        point_batch.append(points)
        id_batch.append(id)
        label_batch.append(labels)
    return (
        id_batch,
        ImageBatch(torch.stack(images)),
        point_batch,
        label_batch,
    )


def train(epochs: int) -> None:
    device = "cuda"
    use_amp = True
    train_dataset = PointDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=1024,
    )
    test_dataset = PointDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=256,
    )
    model = config.net.to(device)
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
    model_loader = config.model_loader
    optimizer = optim.RAdam(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    logs: dict[str, float] = {}
    scaler = GradScaler()

    def train_step() -> None:
        model.train()
        loss_meter = MeanMeter()
        for ids, image_batch, gt_point_batch, gt_label_batch in tqdm(train_loader):
            gt_point_batch = [x.to(device) for x in gt_point_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            image_batch = image_batch.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                netout = model(image_batch)
                _, _, hm_h, hm_w = netout.shape
                gt_hm = config.mkmaps(gt_point_batch, gt_label_batch, w=hm_w, h=hm_h)
                loss = config.hmloss(
                    netout,
                    gt_hm,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())

        logs["train_loss"] = loss_meter.get_value()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        metrics = MeanAveragePrecision(
            iou_threshold=0.3, num_classes=config.num_classes
        )
        for ids, image_batch, gt_point_batch, gt_label_batch in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_point_batch = [x.to(device) for x in gt_point_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            _, _, h, w = image_batch.shape
            netout = model(image_batch)
            _, _, hm_h, hm_w = netout.shape
            gt_hms = config.mkmaps(gt_point_batch, gt_label_batch, w=hm_w, h=hm_h)
            loss = config.hmloss(
                netout,
                gt_hms,
            )
            point_batch, confidence_batch, label_batch = config.to_points(
                netout, h=h, w=w
            )

            loss_meter.update(loss.item())
            for points, gt_points, labels, gt_labels, confidences, image, gt_hm in zip(
                point_batch,
                gt_point_batch,
                label_batch,
                gt_label_batch,
                confidence_batch,
                image_batch,
                gt_hms,
            ):
                ...

        plot = DetectionPlot(image)
        plot.draw_points(points, color="blue", size=4)
        plot.draw_points(gt_points, color="red", size=2)
        plot.save(f"{config.out_dir}/plot-points-.png")

        plot = DetectionPlot(torch.max(gt_hm, dim=0)[0])
        plot.save(f"{config.out_dir}/gt-hm.png")

        logs["test_loss"] = loss_meter.get_value()
        # for k, v in scores.items():
        #     logs[f"score-{k}"] = v
        model_loader.save_if_needed(
            model,
            loss.item(),
        )

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        eval_step()
        train_step()
        log()


if __name__ == "__main__":
    train(1000)
