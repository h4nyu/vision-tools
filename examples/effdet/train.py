import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.metrics import MeanAveragePrecision
from tqdm import tqdm
from object_detection.models.effidet import (
    collate_fn,
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from examples.data import TrainDataset
from object_detection.metrics import MeanPrecition
import torch_optimizer as optim
from examples.effdet import config
from logging import (
    getLogger,
)

logger = getLogger(__name__)


def train(epochs: int) -> None:
    device = torch.device("cuda")
    train_dataset = TrainDataset(
        config.input_size,
        object_count_range=config.object_count_range,
        object_size_range=config.object_size_range,
        num_samples=1024,
    )
    test_dataset = TrainDataset(
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
        eps=1e-8,
        weight_decay=0,
    )
    visualize = Visualize("/store/efficientdet", "test", limit=2)
    get_score = MeanPrecition()
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
    metrics = MeanAveragePrecision(iou_threshold=0.3, num_classes=config.num_classes)

    def train_step() -> None:
        model.train()
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

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        for image_batch, gt_box_batch, gt_label_batch, _ in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            netout = model(image_batch)
            loss, box_loss, label_loss = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = to_boxes(netout)

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
        logger.info(f"{score=}, {scores=}")

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

    for _ in range(epochs):
        train_step()
        eval_step()


if __name__ == "__main__":
    train(10000)
