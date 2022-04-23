from __future__ import annotations

import pathlib
import shutil
from typing import Any, Optional

import albumentations as A
import pandas as pd
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from cytoolz.curried import concat, groupby, map, pipe, sorted, valmap
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_metric_learning.losses import ArcFaceLoss, SubCenterArcFaceLoss
from toolz.curried import filter, map, pipe, topk
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import hflip
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
    merge_rows_by_image_id,
)
from hwad_bench.metrics import MeanAveragePrecisionK
from hwad_bench.scheduler import WarmupReduceLROnPlaetou
from vision_tools.meter import MeanReduceDict
from vision_tools.utils import Checkpoint, ToDevice, seed_everything

from .matchers import MeanEmbeddingMatcher, NearestMatcher


class EnsembleSubmission:
    def __init__(
        self, topk: int = 5, order: str = "desc", strategy: str = "mean"
    ) -> None:
        self.topk = topk
        self.reverse = order == "desc"
        self.strategy = strategy

    def _ensemble(self, rows: list[Submission]) -> Submission:
        image_id = rows[0]["image_id"]
        voting = {}
        distances = pipe(rows, map(lambda x: x["distances"]), concat, list)
        individual_ids = pipe(rows, map(lambda x: x["individual_ids"]), concat, list)
        for dis, id in zip(distances, individual_ids):
            if id not in voting:
                voting[id] = {"total": dis, "count": 1, "best": dis}
            else:
                voting[id] = {
                    "total": voting[id]["total"] + dis,
                    "count": voting[id]["count"] + 1,
                    "best": max(voting[id]["best"], dis),
                }
        merged = pipe(
            voting,
            valmap(lambda x: x["total"] if self.strategy == "mean" else x["best"]),
            lambda x: x.items(),
            sorted(key=lambda x: x[1], reverse=self.reverse),
        )

        res = Submission(
            image_id=image_id,
            individual_ids=pipe(merged, map(lambda x: x[0]), list),
            distances=pipe(merged, map(lambda x: x[1]), list),
        )
        return res

    def __call__(self, rows: list[Submission]) -> list[Submission]:
        groups = groupby(lambda x: x["image_id"], rows)
        res: list[Submission] = []
        for _, group_rows in groups.items():
            res.append(self._ensemble(group_rows))
        return res


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


class EmbNet(nn.Module):
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
        sub_centers: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_supclasses = num_supclasses
        self.sub_centers = sub_centers
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
    return EmbNet(
        name=cfg["name"],
        pretrained=cfg["pretrained"],
        embedding_size=cfg["embedding_size"],
    )


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
    image_dir: str,
    train_rows: list[Annotation],
    body_rows: list[Annotation],
    fin_rows: list[Annotation],
    fold_train: list[dict],
    fold_val: list[dict],
) -> None:
    seed_everything()
    use_amp = cfg["use_amp"]
    eval_interval = cfg["eval_interval"]
    device = cfg["device"]
    accumulate_steps = cfg["accumulate_steps"]
    cfg_name = get_cfg_name(cfg)
    writer = get_writer(cfg)
    min_samples = cfg["min_samples"]
    train_body_rows = filter_rows_by_fold(
        body_rows,
        fold_train,
    )
    train_fin_rows = filter_rows_by_fold(
        fin_rows,
        fold_train,
    )
    val_body_rows = filter_rows_by_fold(
        body_rows,
        fold_val,
    )
    val_fin_rows = filter_rows_by_fold(
        fin_rows,
        fold_val,
    )
    croped_train_rows = train_body_rows + train_fin_rows
    val_rows = merge_rows_by_image_id(
        val_body_rows,
        val_fin_rows,
    )
    croped_train_dataset = HwadCropedDataset(
        rows=croped_train_rows,
        image_dir=image_dir,
        transform=TrainTransform(cfg),
    )
    train_dataset = HwadCropedDataset(
        rows=train_rows,
        image_dir="/app/datasets/hwad-train",
        transform=TrainTransform(cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=filter_in_rows(val_rows, train_rows),
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    train_loader = DataLoader(
        croped_train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    loss_fn = Criterion(
        num_classes=cfg["num_classes"],
        embedding_size=cfg["embedding_size"],
        num_supclasses=cfg["num_supclasses"],
        sub_centers=cfg["sub_centers"],
        alpha=cfg["alpha"],
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
        print(f"Resuming from iteration {saved_state['iteration']}")
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
    model = nn.DataParallel(model).to(device)  # type: ignore
    loss_fn = nn.DataParallel(loss_fn).to(device)  # type: ignore
    print(f"cfg: {cfg_name}")
    print(f"iteration: {iteration}")
    print(f"best_score: {best_score}")
    print(f"score: {score}")
    to_device = ToDevice(device)
    scaler = GradScaler(enabled=use_amp)
    validate_score = cfg["validate_score"]
    for _ in range((cfg["total_steps"] - iteration) // len(train_loader)):
        train_meter = MeanReduceDict()
        for batch, _ in tqdm(train_loader, total=len(train_loader)):
            model.train()
            loss_fn.train()
            image_batch = batch["image_batch"]
            label_batch = batch["label_batch"]
            suplabel_batch = batch["suplabel_batch"]
            with autocast(enabled=use_amp):
                embeddings = model(image_batch)
                loss = (
                    loss_fn(
                        embeddings,
                        label_batch,
                        suplabel_batch,
                    )["loss"]
                    / accumulate_steps
                ).mean()

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
                    for batch, annots in tqdm(val_loader, total=len(val_loader)):
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        suplabel_batch = batch["suplabel_batch"]
                        embeddings = model(image_batch)
                        if validate_score:
                            _, pred_label_batch = matcher(embeddings, k=5)
                            metric.update(
                                pred_label_batch,
                                label_batch,
                            )
                            loss = loss_fn(
                                embeddings,
                                label_batch,
                                suplabel_batch,
                            )["loss"].mean()
                            val_meter.update({"loss": loss.item()})
                for k, v in train_meter.value.items():
                    writer.add_scalar(f"train/{k}", v, iteration)

                for k, v in val_meter.value.items():
                    writer.add_scalar(f"val/{k}", v, iteration)
                writer.add_scalar(
                    f"val/loss", val_meter.value.get("loss", 0.0), iteration
                )
                writer.add_scalar(
                    f"train/lr", [x["lr"] for x in optimizer.param_groups][0], iteration
                )
                if validate_score:
                    score, _ = metric.value
                    writer.add_scalar(f"val/score", score, iteration)
                train_meter.reset()
                val_meter.reset()

                if validate_score and (score > best_score):
                    best_score = score
                    checkpoint.save(
                        {
                            "model": model.module.state_dict(),  # type: ignore
                            "loss_fn": loss_fn.module.state_dict(),  # type: ignore
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "score": score,
                            "best_score": best_score,
                        },
                        target="best_score",
                    )
                checkpoint.save(
                    {
                        "model": model.module.state_dict(),  # type: ignore
                        "loss_fn": loss_fn.module.state_dict(),  # type: ignore
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration,
                        "best_score": best_score,
                    },
                    target="latest",
                )
                metric.reset()
                val_meter.reset()
            iteration += 1
        writer.flush()


@torch.no_grad()
def cv_registry(
    cfg: dict,
    image_dir: str,
    body_rows: list[Annotation],
    fin_rows: list[Annotation],
    fold_train: list[dict],
    fold_val: list[dict],
) -> MeanEmbeddingMatcher:
    seed_everything()
    device = cfg["device"]
    writer = get_writer(cfg)
    model = get_model(cfg).to(device)
    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load(cfg["resume"])
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

    train_body_rows = filter_rows_by_fold(
        body_rows,
        fold_train,
    )
    train_fin_rows = filter_rows_by_fold(
        fin_rows,
        fold_train,
    )
    reg_rows = train_body_rows + train_fin_rows
    reg_dataset = HwadCropedDataset(
        rows=reg_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    reg_loader = DataLoader(
        reg_dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    to_device = ToDevice(device)
    model.eval()
    matcher = MeanEmbeddingMatcher()

    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
        embeddings = model(hflip(image_batch))
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    return matcher


@torch.no_grad()
def registry(
    cfg: dict,
    image_dir: str,
    body_rows: list[Annotation],
    fin_rows: list[Annotation],
) -> MeanEmbeddingMatcher:
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
    reg_rows = body_rows + body_rows
    transform = Transform(cfg)
    reg_dataset = HwadCropedDataset(
        rows=reg_rows,
        image_dir=image_dir,
        transform=transform,
    )
    hfliped_dataset = HwadCropedDataset(
        rows=reg_rows,
        image_dir=image_dir,
        transform=A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                transform,
            ]
        ),
    )
    reg_loader: DataLoader = DataLoader(
        ConcatDataset([reg_dataset, hfliped_dataset]),
        batch_size=cfg["registry_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    to_device = ToDevice(device)
    model.eval()
    matcher = MeanEmbeddingMatcher()
    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    return matcher


@torch.no_grad()
def cv_evaluate(
    cfg: dict,
    image_dir: str,
    body_rows: list[Annotation],
    fin_rows: list[Annotation],
    fold_train: list[dict],
    fold_val: list[dict],
    matcher: MeanEmbeddingMatcher,
) -> list[Submission]:
    seed_everything()
    device = cfg["device"]
    writer = get_writer(cfg)
    model = get_model(cfg).to(device)
    checkpoint = get_checkpoint(cfg)
    saved_state = checkpoint.load(cfg["resume"])
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

    train_body_rows = filter_rows_by_fold(
        body_rows,
        fold_train,
    )
    train_fin_rows = filter_rows_by_fold(
        fin_rows,
        fold_train,
    )
    val_body_rows = filter_rows_by_fold(
        body_rows,
        fold_val,
    )
    val_fin_rows = filter_rows_by_fold(
        fin_rows,
        fold_val,
    )
    reg_rows = train_body_rows + train_fin_rows
    val_rows = val_body_rows + val_fin_rows
    reg_dataset = HwadCropedDataset(
        rows=reg_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_rows,
        image_dir=image_dir,
        transform=Transform(cfg),
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
    ensemble = EnsembleSubmission()

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
                image_id=annot["image_file"].split(".")[0],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
        embeddings = model(hflip(image_batch))
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
                image_id=annot["image_file"].split(".")[0],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
    rows = ensemble(rows)
    return rows


@torch.no_grad()
def inference(
    cfg: dict,
    train_body_rows: list[Annotation],
    train_fin_rows: list[Annotation],
    test_body_rows: list[Annotation],
    test_fin_rows: list[Annotation],
    image_dir: str,
    matcher: MeanEmbeddingMatcher,
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

    id_map = pipe(
        train_fin_rows + train_body_rows,
        map(lambda x: (x["label"], x["individual_id"])),
        dict,
    )

    test_rows = test_body_rows + test_fin_rows
    transform = Transform(cfg)
    test_dataset = HwadCropedDataset(
        rows=test_rows,
        image_dir=image_dir,
        transform=transform,
    )
    hfliped_dataset = HwadCropedDataset(
        rows=test_rows,
        image_dir=image_dir,
        transform=A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                transform,
            ]
        ),
    )
    test_loader: DataLoader = DataLoader(
        ConcatDataset([test_dataset, hfliped_dataset]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    to_device = ToDevice(cfg["device"])
    model.eval()
    rows: list[Submission] = []
    for batch, batch_annots in tqdm(test_loader, total=len(test_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        topk_distance, pred_label_batch = matcher(embeddings, k=50)
        for pred_topk, annot, distances in zip(
            pred_label_batch.tolist(), batch_annots, topk_distance.tolist()
        ):
            individual_ids = pipe(
                pred_topk,
                map(lambda x: id_map[x]),
                list,
            )
            row = Submission(
                image_id=annot["image_file"].split(".")[0],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
    return rows


def search_threshold(
    fold_train: list[dict],
    fold_val: list[dict],
    submissions: list[Submission],
    thresholds: list[float],
) -> list[dict]:

    train_individual_ids = pipe(
        fold_train,
        map(lambda x: x["individual_id"]),
        set,
    )
    val_individual_ids = pipe(
        fold_val,
        map(lambda x: x["individual_id"]),
        set,
    )
    print(len(fold_train), len(fold_val))
    new_individual_ids = val_individual_ids - train_individual_ids

    val_rows = pipe(
        fold_val,
        map(
            lambda x: {
                **x,
                "individual_id": "new_individual"
                if x["individual_id"] in new_individual_ids
                else x["individual_id"],
            }
        ),
        list,
    )

    count = 0
    val_annot_map = pipe(
        val_rows,
        map(
            lambda x: (x["image"].split(".")[0], x),
        ),
        dict,
    )
    all_individual_ids = train_individual_ids | set(["new_individual"])
    label_map = pipe(
        all_individual_ids,
        sorted,
        enumerate,
        map(lambda x: (x[1], x[0])),
        dict,
    )
    results = []
    for thr in tqdm(thresholds):
        metric = MeanAveragePrecisionK()
        thr_submissions = add_new_individual(submissions, thr)
        for sub in thr_submissions:
            labels_at_k = (
                pipe(
                    sub["individual_ids"],
                    map(lambda x: label_map[x]),
                    list,
                    torch.tensor,
                )
                .long()
                .unsqueeze(0)
            )
            annot = val_annot_map[sub["image_id"]]
            labels = pipe(
                [annot["individual_id"]],
                map(lambda x: label_map[x]),
                list,
                torch.tensor,
            ).long()
            metric.update(labels_at_k, labels)
        score, _ = metric.value
        results.append(
            {
                "threshold": thr,
                "score": score,
            }
        )
        print(f"Threshold: {thr} Score: {score}")
    return results


def post_process(
    cfg: dict,
    rows: list[Submission],
    train_rows: list[Annotation],
) -> list[Submission]:
    individual_id_spicies = pipe(
        train_rows,
        map(lambda x: (x["individual_id"], x["species"])),
        dict,
    )
    # for row in tqdm(rows):
    #     top1 = row["individual_ids"][0]
    #     top1_species = individual_id_spicies[top1]
    #     matched_index = pipe(
    #         range(len(row["individual_ids"])),
    #         filter(lambda x: individual_id_spicies[row["individual_ids"][x]] == top1_species),
    #         list,
    #     )[:5]
    #     individual_ids = pipe(
    #         matched_index,
    #         map(lambda x: row["individual_ids"][x]),
    #         list,
    #     )
    #     distances = pipe(
    #         matched_index,
    #         map(lambda x: row["distances"][x]),
    #         list,
    #     )
    #     row["individual_ids"] = individual_ids
    #     row["distances"] = distances

    ensemble = EnsembleSubmission(strategy=cfg.get("ensemble_strategy", "mean"))
    rows = ensemble(rows)
    rows = add_new_individual(rows, cfg.get("threshold", 0.5))
    return rows


def save_submission(
    submissions: list[Submission],
    output_path: str,
) -> list[dict]:
    rows: list[dict] = []
    for sub in submissions:
        image_id = sub["image_id"]
        row = {
            "image": f"{image_id}.jpg",
            "predictions": " ".join(sub["individual_ids"]),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return rows


def preview(
    submissions: list[Submission],
    train_rows: list[Annotation],
    output_path: str,
) -> None:
    rows: list[dict] = []
    id_image_id = pipe(
        train_rows,
        map(lambda x: (x["individual_id"], x["image_file"].split(".")[0])),
        dict,
    )

    for sub in submissions:
        image_id = sub["image_id"]
        query = pathlib.Path("/app/datasets/hwad-test-croped", f"{image_id}.body.jpg")
        shutil.copy(query, pathlib.Path(output_path, f"{image_id}.jpg"))
        for individual_id in sub["individual_ids"]:
            if individual_id == "new_individual":
                continue
            shutil.copy(
                pathlib.Path(
                    "/app/datasets/hwad-train-croped",
                    f"{id_image_id[individual_id]}.body.jpg",
                ),
                pathlib.Path(output_path, f"{image_id}.top.jpg"),
            )


def ensemble_submissions(
    submission0: list[Submission],
    submission1: list[Submission],
    submission2: list[Submission],
    train_rows: list[Annotation],
    threshold: float,
    output_path: str,
) -> list[Submission]:
    rows = submission0 + submission1 + submission2
    print(len(rows))
    individual_id_spicies = pipe(
        train_rows,
        map(lambda x: (x["individual_id"], x["species"])),
        dict,
    )
    for row in tqdm(rows):
        row["individual_ids"] = row["individual_ids"][:5]
        row["distances"] = row["distances"][:5]
    ensemble = EnsembleSubmission()
    rows = ensemble(rows)
    count = 0
    for row in rows:
        spicies = pipe(
            row["individual_ids"],
            map(lambda x: individual_id_spicies.get(x, None)),
            list,
        )[:5]
        if len(set(spicies)) > 3:
            print(spicies)
            count += 1
    rows = add_new_individual(rows, threshold)
    save_submission(rows, output_path)
    return rows


def ensemble_files(
    paths: list[str],
    output_path: str,
) -> list[Submission]:
    rows: list[Submission] = []
    distances = pipe(
        range(5),
        map(lambda x: x + 1.0),
        list,
        reversed,
        list,
    )

    for path in paths:
        df = pd.read_csv(path)
        for row in df.itertuples():
            rows.append(
                Submission(
                    {
                        "image_id": row.image.split(".")[0],
                        "individual_ids": row.predictions.split(" ")[:5],
                        "distances": distances,
                    }
                )
            )
    ensemble = EnsembleSubmission()
    rows = ensemble(rows)
    save_submission(rows, output_path)
    return rows
