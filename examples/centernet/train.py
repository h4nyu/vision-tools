import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Criterion,
    ToBoxes,
)
from object_detection.entities import (
    yolo_to_pascal,
)
import torch_optimizer as optim
from object_detection.models.mkmaps import (
    MkGaussianMaps,
    MkCenterBoxMaps,
)
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from examples.data import TrainDataset
from object_detection.metrics import MeanAveragePrecision
from examples.centernet import config as cfg


def train(epochs: int) -> None:
    device = "cuda"
    use_amp = True
    train_dataset = TrainDataset(
        cfg.input_size,
        object_count_range=cfg.object_count_range,
        object_size_range=cfg.object_size_range,
        num_samples=1024,
    )
    test_dataset = TrainDataset(
        cfg.input_size,
        object_count_range=cfg.object_count_range,
        object_size_range=cfg.object_size_range,
        num_samples=256,
    )
    backbone = EfficientNetBackbone(1, out_channels=cfg.channels, pretrained=True)
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
    visualize = Visualize(cfg.out_dir, "test", limit=2)
    metrics = MeanAveragePrecision(iou_threshold=0.3, num_classes=cfg.num_classes)

    model_loader = ModelLoader(
        out_dir=cfg.out_dir,
        key=cfg.metric[0],
        best_watcher=BestWatcher(mode=cfg.metric[1]),
    )
    to_boxes = ToBoxes(threshold=cfg.to_boxes_threshold)
    get_score = lambda *args: MeanAveragePrecision(
        iou_threshold=0.3, num_classes=cfg.num_classes
    )(*args)[0]
    scaler = GradScaler()

    def train_step() -> None:
        model.train()
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(train_loader):
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            image_batch = image_batch.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                netout = model(image_batch)
                loss, hm_loss, bm_loss, _ = criterion(
                    image_batch, netout, gt_box_batch, gt_label_batch
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        for ids, image_batch, gt_box_batch, gt_label_batch in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            _, _, h, w = image_batch.shape
            netout = model(image_batch)
            loss, hm_loss, bm_loss, gt_hms = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = to_boxes(netout)
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
        print(f"{score=}, {scores=}")
        metrics.reset()

        model_loader.save_if_needed(
            model,
            score,
        )

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()
        eval_step()


if __name__ == "__main__":
    train(1000)
