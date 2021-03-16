import torch, tqdm, os
from typing import Any
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from logging import (
    getLogger,
)
from vnet import (
    Image,
    ImageBatch,
    Boxes,
    Labels,
    Boxes,
)
from vnet.meters import MeanMeter
import torch_optimizer as optim
from bench.kuzushiji.effdet.config import Config
from bench.kuzushiji.data import (
    KuzushijiDataset,
    read_train_rows,
    train_transforms,
    kfold,
    inv_normalize,
)
from bench.kuzushiji.metrics import Metrics
from vnet.utils import DetectionPlot

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
    config = Config()
    device = config.device
    rows = read_train_rows(config.root_dir)
    train_rows, test_rows = kfold(rows, config.n_splits)
    train_dataset = KuzushijiDataset(
        rows=train_rows,
        transforms=train_transforms,
    )
    test_dataset = KuzushijiDataset(
        rows=test_rows,
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=min(os.cpu_count() or 1, config.batch_size),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 2,
        num_workers=min(os.cpu_count() or 1, config.batch_size * 2),
        shuffle=False,
    )
    optimizer = config.optimizer
    to_boxes = config.to_boxes
    model = config.model
    model_loader = config.model_loader
    criterion = config.criterion
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
        ) in tqdm.tqdm(train_loader):
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
        metrics = Metrics()
        for image_batch, gt_box_batch, gt_label_batch, _ in tqdm.tqdm(test_loader):
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

            for boxes, gt_boxes, labels, gt_labels, image in zip(
                box_batch, gt_box_batch, label_batch, gt_label_batch, image_batch
            ):
                points = config.to_points(boxes)
                metrics.add(
                    points=points,
                    labels=labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )
            plot = DetectionPlot(inv_normalize(image))
            plot.draw_points(points, color="blue")
            plot.draw_boxes(boxes, color="red")
            plot.save(os.path.join(config.out_dir, "test.png"))

        score = metrics()
        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"] = score

        model_loader.save_if_needed(model, logs[model_loader.key])

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()
        eval_step()
        log()


if __name__ == "__main__":
    train(10000)
