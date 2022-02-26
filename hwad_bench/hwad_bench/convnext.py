from __future__ import annotations

import timm
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from toolz.curried import map, pipe
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def get_writer(cfg: dict) -> SummaryWriter:
    model_name = get_model_name(cfg)
    writer_name = pipe(
        [
            model_name,
            cfg["lr"],
        ],
        map(str),
        "-".join,
    )
    return SummaryWriter(
        f"runs/{writer_name}",
    )


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
    use_amp = train_cfg["use_amp"]
    eval_interval = train_cfg["eval_interval"]
    device = model_cfg["device"]
    writer = get_writer({**train_cfg, **model_cfg})

    loss_fn = ArcFaceLoss(
        num_classes=dataset_cfg["num_classes"],
        embedding_size=model_cfg["embedding_size"],
    )
    model = get_model(model_cfg).to(device)
    optimizer = Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=train_cfg["lr"],
    )

    checkpoint = get_checkpoint(model_cfg)
    saved_state = checkpoint.load(train_cfg["resume"])
    iteration = 0
    best_score = float("inf")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        loss_fn.load_state_dict(saved_state["loss_fn"])
        optimizer.load_state_dict(saved_state["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        iteration = saved_state.get("iteration", 0)
        best_score = saved_state.get("score", float("inf"))
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
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    epoch_size = len(train_loader)
    to_device = ToDevice(model_cfg["device"])
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(train_cfg["epochs"]):
        train_meter = MeanReduceDict()
        model.train()
        for batch in tqdm(train_loader, total=epoch_size):
            iteration += 1
            batch = to_device(**batch)
            image_batch = batch["image_batch"]
            label_batch = batch["label_batch"]

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                embeddings = model(image_batch)
                loss = loss_fn(
                    embeddings,
                    label_batch,
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()

            train_meter.update({"loss": loss.item()})

            if iteration % eval_interval == 0:
                model.eval()
                val_meter = MeanReduceDict()
                with torch.no_grad():
                    for batch in tqdm(val_loader, total=len(val_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)
                        loss = loss_fn(
                            embeddings,
                            label_batch,
                        )
                        val_meter.update({"loss": loss.item()})

                for k, v in train_meter.value.items():
                    writer.add_scalar(f"train/{k}", v, iteration)

                for k, v in val_meter.value.items():
                    writer.add_scalar(f"val/{k}", v, iteration)
                score = val_meter.value["loss"]
                if score < best_score:
                    best_score = score
                    checkpoint.save(
                        {
                            "model": model.state_dict(),
                            "loss_fn": loss_fn.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "score": score,
                        },
                        target="best",
                    )
                checkpoint.save(
                    {
                        "model": model.state_dict(),
                        "loss_fn": loss_fn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration,
                        "score": score,
                    },
                    target="latest",
                )
                train_meter.reset()
                val_meter.reset()

        writer.flush()
