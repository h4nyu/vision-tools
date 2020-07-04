import torch
from torch.utils.data import DataLoader
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Trainer,
    Criterion,
)
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.model_loader import ModelLoader
from object_detection.data.object import ObjectDataset
from object_detection.metrics import MeanPrecition
from object_detection.meters import BestWatcher
from logging import getLogger, StreamHandler, Formatter, INFO, FileHandler

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(INFO)
handler_format = Formatter("%(asctime)s|%(name)s|%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

train_dataset = ObjectDataset(
    (512, 512), object_count_range=(1, 50), object_size_range=(32, 128), num_samples=1024
)
test_dataset = ObjectDataset(
    (512, 512), object_count_range=(1, 50), object_size_range=(32, 128), num_samples=256
)
channels = 256
backbone = ResNetBackbone("resnet50", out_channels=channels)
model = CenterNet(channels=channels, backbone=backbone, out_idx=5, depth=1)
model_loader = ModelLoader("/store/centernet")
criterion = Criterion(sizemap_weight=1.0, sigma=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
visualize = Visualize("/store/centernet", "test", limit=2)
best_watcher = BestWatcher(mode="max")
trainer = Trainer(
    model=model,
    train_loader=DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True
    ),
    test_loader=DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True
    ),
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    criterion=criterion,
    best_watcher=best_watcher,
    device="cuda",
    get_score=MeanPrecition(),
)
trainer.train(500)
