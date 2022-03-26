from __future__ import annotations

from typing import Any, Optional

import timm
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_metric_learning.losses import ArcFaceLoss
from toolz.curried import filter, map, pipe, topk
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hwad_bench.data import (
    Annotation,
    HwadCropedDataset,
    Submission,
    TrainTransform,
    Transform,
    add_new_individual,
    collate_fn,
    filter_in_rows,
    filter_rows_by_fold,
)
from hwad_bench.metrics import MeanAveragePrecisionK
from hwad_bench.scheduler import WarmupReduceLROnPlaetou
from vision_tools.meter import MeanReduceDict
from vision_tools.utils import Checkpoint, ToDevice, seed_everything

from .matchers import MeanEmbeddingMatcher, NearestMatcher


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ConvNeXt(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained, num_classes=embedding_size)
        self.embedding = nn.LazyLinear(embedding_size)
        self.pooling = GeM()

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward_features(x)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding


class EfficientNet(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained, num_classes=embedding_size)
        self.embedding = nn.Linear(self.model.num_features, embedding_size)
        self.pooling = GeM()

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward_features(x)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding


class Criterion(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        num_supclasses: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_supclasses = num_supclasses
        self.arcface = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
        )
        self.alpha = alpha
        self.supcls_fc = nn.Linear(embedding_size, self.num_supclasses)

    def forward(
        self, embeddings: Tensor, labels: Tensor, suplabels: Tensor
    ) -> dict[str, Tensor]:
        cls_loss = self.arcface(embeddings, labels)
        supcls_loss = F.cross_entropy(self.supcls_fc(embeddings), suplabels)
        loss = cls_loss + self.alpha * supcls_loss
        return {"loss": loss, "cls_loss": cls_loss, "supcls_loss": supcls_loss}


def get_cfg_name(cfg: dict) -> str:
    keys = [
        "version",
        "fold",
    ]
    return "_".join([str(cfg[k]) for k in keys])


def get_model(cfg: dict) -> Any:
    if cfg["name"] in ["convnext_small"]:
        return ConvNeXt(
            name=cfg["name"],
            pretrained=cfg["pretrained"],
            embedding_size=cfg["embedding_size"],
        )
    if "efficientnet" in cfg["name"]:
        return EfficientNet(
            name=cfg["name"],
            pretrained=cfg["pretrained"],
            embedding_size=cfg["embedding_size"],
        )
    raise Exception(f"Unknown model name: {cfg['name']}")


def get_checkpoint(cfg: dict) -> Checkpoint:
    model = Checkpoint(
        root_dir=get_cfg_name(cfg),
    )
    return model


def get_writer(cfg: dict) -> SummaryWriter:
    return SummaryWriter(
        f"/app/hwad_bench/pipeline/runs/{get_cfg_name(cfg)}",
    )


def train(
    cfg: dict,
    train_rows: list[Annotation],
    val_rows: list[Annotation],
    image_dir: str,
) -> None:
    seed_everything()
    use_amp = cfg["use_amp"]
    eval_interval = cfg["eval_interval"]
    device = cfg["device"]
    accumulate_steps = cfg["accumulate_steps"]
    cfg_name = get_cfg_name(cfg)
    writer = get_writer(cfg)
    cleaned_rows = pipe(
        train_rows,
        filter(lambda x: x["individual_samples"] >= cfg["min_samples"]),
        list,
    )
    train_dataset = HwadCropedDataset(
        rows=cleaned_rows,
        image_dir=image_dir,
        transform=TrainTransform(cfg),
    )
    reg_dataset = HwadCropedDataset(
        rows=train_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=filter_in_rows(val_rows, train_rows),
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    reg_loader = DataLoader(
        reg_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    # TODO num_class
    loss_fn = ArcFaceLoss(
        num_classes=cfg["num_classes"],
        embedding_size=cfg["embedding_size"],
    )
    model = get_model(cfg)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load(cfg["resume"])
    iteration = 0
    score = 0.0
    best_score = 0.0
    scheduler = (
        OneCycleLR(
            optimizer,
            total_steps=cfg["total_steps"],
            max_lr=cfg["lr"],
            pct_start=cfg["warmup_steps"] / cfg["total_steps"],
            final_div_factor=cfg["final_div_factor"],
        )
        if cfg["use_scheduler"]
        else None
    )
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        loss_fn.load_state_dict(saved_state["loss_fn"])
        optimizer.load_state_dict(saved_state["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        iteration = saved_state.get("iteration", 0)
        best_score = saved_state.get("best_score", 0.0)
        socore = saved_state.get("score", 0.0)
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    print(f"cfg: {cfg_name}")
    print(f"iteration: {iteration}")
    print(f"best_score: {best_score}")
    to_device = ToDevice(device)
    scaler = GradScaler(enabled=use_amp)
    for _ in range((cfg["total_steps"] - iteration) // len(train_loader)):
        train_meter = MeanReduceDict()
        for batch, _ in tqdm(train_loader, total=len(train_loader)):
            iteration += 1
            model.train()
            loss_fn.train()
            batch = to_device(**batch)
            image_batch = batch["image_batch"]
            label_batch = batch["label_batch"]
            with autocast(enabled=use_amp):
                embeddings = model(image_batch)
                loss = (
                    loss_fn(
                        embeddings,
                        label_batch,
                    )
                    / accumulate_steps
                )

                scaler.scale(loss).backward()
                if iteration % accumulate_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step(iteration)
            train_meter.update({"loss": loss.item() * accumulate_steps})

            if iteration % eval_interval == 0:
                model.eval()
                loss_fn.eval()
                val_meter = MeanReduceDict()
                metric = MeanAveragePrecisionK()
                matcher = MeanEmbeddingMatcher()
                with torch.no_grad():

                    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)
                        matcher.update(embeddings, label_batch)
                    matcher.create_index()
                    for batch, annots in tqdm(val_loader, total=len(val_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)
                        _, pred_label_batch = matcher(embeddings, k=5)
                        loss = loss_fn(
                            embeddings,
                            label_batch,
                        )
                        metric.update(
                            pred_label_batch,
                            label_batch,
                        )
                        val_meter.update({"loss": loss.item()})
                score, _ = metric.value
                for k, v in train_meter.value.items():
                    writer.add_scalar(f"train/{k}", v, iteration)

                for k, v in val_meter.value.items():
                    writer.add_scalar(f"val/{k}", v, iteration)
                writer.add_scalar(f"val/loss", val_meter.value["loss"], iteration)
                writer.add_scalar(
                    f"train/lr", [x["lr"] for x in optimizer.param_groups][0], iteration
                )
                writer.add_scalar(f"val/score", score, iteration)
                train_meter.reset()
                val_meter.reset()

            if score < best_score:
                best_score = score
                checkpoint.save(
                    {
                        "model": model.state_dict(),
                        "loss_fn": loss_fn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration,
                        "score": score,
                        "best_score": best_score,
                    },
                    target="best_score",
                )
                checkpoint.save(
                    {
                        "model": model.state_dict(),
                        "loss_fn": loss_fn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration,
                        "best_score": best_score,
                    },
                    target="latest",
                )
                metric.reset()
                val_meter.reset()
        writer.flush()


@torch.no_grad()
def eval_epoch(
    cfg: dict,
    train_rows: list[Annotation],
    val_rows: list[Annotation],
    image_dir: str,
) -> None:
    seed_everything()
    eval_interval = cfg["eval_interval"]
    device = cfg["device"]
    cfg_name = get_cfg_name(cfg)
    writer = get_writer(cfg)
    reg_dataset = HwadCropedDataset(
        rows=train_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=filter_in_rows(val_rows, train_rows),
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"] // 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    reg_loader = DataLoader(
        reg_dataset,
        batch_size=cfg["batch_size"] // 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    model = get_model(cfg).to(device)
    loss_fn = ArcFaceLoss(
        num_classes=cfg["num_classes"],
        embedding_size=cfg["embedding_size"],
    ).to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load(cfg["resume"])
    iteration = 0
    best_score = 0.0
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        loss_fn.load_state_dict(saved_state["loss_fn"])
        optimizer.load_state_dict(saved_state["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        iteration = saved_state.get("iteration", 0)
        best_score = saved_state.get("best_score", 0.0)
    print(f"cfg: {cfg_name}")
    print(f"iteration: {iteration}")
    print(f"best_score: {best_score}")
    to_device = ToDevice(device)

    model.eval()
    metric = MeanAveragePrecisionK()
    val_meter = MeanReduceDict()
    matcher = MeanEmbeddingMatcher()
    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    for batch, annots in tqdm(val_loader, total=len(val_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        _, pred_label_batch = matcher(embeddings, k=5)
        metric.update(
            pred_label_batch,
            label_batch,
        )
    score, _ = metric.value
    writer.add_scalar(f"val/score", score, iteration)
    if score < best_score:
        best_score = score
        checkpoint.save(
            {
                "model": model.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "score": score,
                "best_score": best_score,
            },
            target="best_score",
        )
    print(f"score: {score}")
    metric.reset()
    val_meter.reset()


@torch.no_grad()
def evaluate(
    cfg: dict,
    fold: int,
    train_rows: list[Annotation],
    val_rows: list[Annotation],
    image_dir: str,
) -> list[Submission]:
    seed_everything()
    device = cfg["device"]
    writer = get_writer(cfg)
    model = get_model(cfg).to(device)
    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load("latest")
    iteration = 0
    best_score = 0.0
    best_loss = float("inf")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        iteration = saved_state.get("iteration", 0)
        best_loss = saved_state.get("best_loss", float("inf"))
        best_score = saved_state.get("best_score", 0.0)
    print(f"Loaded checkpoint from iteration {iteration}")
    print(f"Best score: {best_score}")
    print(f"Best loss: {best_loss}")

    reg_dataset = HwadCropedDataset(
        rows=train_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )

    reg_loader = DataLoader(
        reg_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    to_device = ToDevice(device)
    model.eval()
    val_meter = MeanReduceDict()
    matcher = NearestMatcher()

    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    rows: list[Submission] = []
    for batch, batch_annots in tqdm(val_loader, total=len(val_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        topk_distance, pred_label_batch = matcher(embeddings, k=5)
        for pred_topk, annot, distances in zip(
            pred_label_batch.tolist(), batch_annots, topk_distance.tolist()
        ):
            individual_ids = pipe(
                pred_topk,
                map(lambda x: reg_dataset.id_map[x]),
                list,
            )
            row = Submission(
                image_file=annot["image_file"],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
    return rows


@torch.no_grad()
def inference(
    cfg: dict,
    train_rows: list[Annotation],
    test_rows: list[Annotation],
    search_thresholds: list[dict],
    threshold: Optional[float],
    train_image_dir: str,
    test_image_dir: str,
) -> list[Submission]:
    seed_everything()
    device = cfg["device"]
    writer = get_writer(cfg)
    model = get_model(cfg).to(device)
    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load("latest")
    iteration = 0
    best_score = 0.0
    best_loss = float("inf")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        iteration = saved_state.get("iteration", 0)
        best_loss = saved_state.get("best_loss", float("inf"))
        best_score = saved_state.get("best_score", 0.0)
    print(f"Loaded checkpoint from iteration {iteration}")
    print(f"Best score: {best_score}")
    print(f"Best loss: {best_loss}")

    train_dataset = HwadCropedDataset(
        rows=train_rows,
        image_dir=train_image_dir,
        transform=Transform(cfg),
    )
    test_dataset = HwadCropedDataset(
        rows=test_rows,
        image_dir=test_image_dir,
        transform=Transform(cfg),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    to_device = ToDevice(cfg["device"])
    _threshold = (
        threshold
        or topk(1, search_thresholds, key=lambda x: x["score"])[0]["threshold"]
    )
    print(f"Threshold: {_threshold}")
    model.eval()
    matcher = NearestMatcher()
    for batch, batch_annot in tqdm(train_loader, total=len(train_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    rows: list[Submission] = []
    for batch, batch_annots in tqdm(test_loader, total=len(test_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        topk_distance, pred_label_batch = matcher(embeddings, k=5)
        for pred_topk, annot, distances in zip(
            pred_label_batch.tolist(), batch_annots, topk_distance.tolist()
        ):
            individual_ids = pipe(
                pred_topk,
                map(lambda x: train_dataset.id_map[x]),
                list,
            )
            row = Submission(
                image_file=annot["image_file"],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
    rows = add_new_individual(rows, _threshold)
    return rows
