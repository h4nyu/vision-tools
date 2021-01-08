import torch
from torch.utils.data import DataLoader
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Trainer,
    Criterion,
    ToBoxes,
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
from object_detection.metrics import MeanPrecition
from examples.centernet import config as cfg


def train(epochs: int) -> None:
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
    )
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

    model_loader = ModelLoader(
        out_dir=cfg.out_dir,
        key=cfg.metric[0],
        best_watcher=BestWatcher(mode=cfg.metric[1]),
    )
    to_boxes = ToBoxes(threshold=cfg.to_boxes_threshold)
    get_score = MeanPrecition()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        criterion=criterion,
        device="cuda",
        get_score=get_score,
        to_boxes=to_boxes,
        use_amp=True,
    )
    trainer(epochs)


if __name__ == "__main__":
    train(1000)
