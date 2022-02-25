from __future__ import annotations

import timm
from pytorch_metric_learning.losses import ArcFaceLoss
from toolz.curried import map, pipe
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hwad_bench.data import (
    Annotation,
    HwadCropedDataset,
    TrainTransform,
    Transform,
    collate_fn,
    filter_annotations_by_fold,
)
from vision_tools.meter import MeanReduceDict
from vision_tools.utils import Checkpoint, ToDevice, seed_everything


class ConvNeXt(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained, num_classes=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        out = F.normalize(out, p=2, dim=1)
        return out


def get_model_name(cfg: dict) -> str:
    return pipe(
        [
            cfg["name"],
            cfg["fold"],
            cfg["embedding_size"],
            cfg["pretrained"],
        ],
        map(str),
        "-".join,
    )


def get_model(cfg: dict) -> ConvNeXt:
    model = ConvNeXt(
        name=cfg["name"],
        pretrained=cfg["pretrained"],
        embedding_size=cfg["embedding_size"],
    )
    return model


def get_checkpoint(cfg: dict) -> Checkpoint:
    model = Checkpoint(
        root_dir=get_model_name(cfg),
    )
    return model


def train(
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    fold: int,
    annotations: list[Annotation],
    fold_train: list[dict],
    fold_val: list[dict],
    image_dir: str,
) -> None:
    seed_everything()
    loss_fn = ArcFaceLoss(
        num_classes=dataset_cfg["num_classes"],
        embedding_size=model_cfg["embedding_size"],
    )
    model = get_model(model_cfg)
    checkpoint = get_checkpoint(model_cfg)
    saved_state = checkpoint.load("best")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
    train_annots = filter_annotations_by_fold(
        annotations, fold_train, min_samples=dataset_cfg["min_samples"]
    )
    val_annots = filter_annotations_by_fold(
        annotations, fold_val, min_samples=dataset_cfg["min_samples"]
    )
    train_dataset = HwadCropedDataset(
        rows=train_annots,
        image_dir=image_dir,
        transform=TrainTransform(dataset_cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_annots,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    epoch_size = len(train_loader)
    to_device = ToDevice(model_cfg["device"])
    for epoch in range(train_cfg["epochs"]):
        train_meter = MeanReduceDict()
        for batch in tqdm(train_loader, total=epoch_size):
            batch = to_device(**batch)
