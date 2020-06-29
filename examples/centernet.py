import torch
from torch.utils.data import DataLoader
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Trainer,
    Criterion,
)
from object_detection.model_loader import ModelLoader
from object_detection.data.object import ObjectDataset
from logging import getLogger, StreamHandler, Formatter, INFO, FileHandler

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(INFO)
handler_format = Formatter("%(asctime)s|%(name)s|%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

train_dataset = ObjectDataset(
    (256, 256), object_count_range=(1, 20), object_size_range=(32, 64), num_samples=256
)
test_dataset = ObjectDataset(
    (256, 256), object_count_range=(1, 20), object_size_range=(32, 64), num_samples=8
)
model = CenterNet(channels=128)
model_loader = ModelLoader("/store/centernet", model=model)
criterion = Criterion(sizemap_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(
    DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=8, num_workers=4, shuffle=True
    ),
    DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=8, num_workers=4, shuffle=True
    ),
    model_loader,
    optimizer,
    Visualize("/store/centernet", "test", limit=2),
    "cuda",
    criterion=criterion,
)
trainer.train(500)
