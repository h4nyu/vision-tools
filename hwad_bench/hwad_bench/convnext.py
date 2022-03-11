from __future__ import annotations

import timm
import torch
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
    filter_annotations_by_fold,
    filter_in_annotations,
)
from hwad_bench.metrics import MeanAveragePrecisionK
from vision_tools.meter import MeanReduceDict
from vision_tools.utils import Checkpoint, ToDevice, seed_everything

from .matchers import MeanEmbeddingMatcher


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
            "fold",
            cfg["fold"],
            cfg["name"],
            cfg["embedding_size"],
            cfg["pretrained"],
            cfg["version"],
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
            cfg["random_resized_crop_p"],
            cfg["min_samples"],
            cfg["batch_size"],
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
    train_annotations: list[Annotation],
    val_annotations: list[Annotation],
    image_dir: str,
) -> None:
    seed_everything()
    use_amp = train_cfg["use_amp"]
    eval_interval = train_cfg["eval_interval"]
    device = model_cfg["device"]
    writer = get_writer({**train_cfg, **model_cfg, **dataset_cfg})
    cleaned_annoations = pipe(
        train_annotations,
        filter(lambda x: x["individual_samples"] >= train_cfg["min_samples"]),
        list,
    )
    train_dataset = HwadCropedDataset(
        rows=cleaned_annoations,
        image_dir=image_dir,
        transform=TrainTransform(dataset_cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_annotations,
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

    # TODO num_class
    loss_fn = ArcFaceLoss(
        num_classes=train_cfg["num_classes"],
        embedding_size=model_cfg["embedding_size"],
    ).to(device)
    model = get_model(model_cfg).to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint = get_checkpoint(model_cfg)
    saved_state = checkpoint.load(train_cfg["resume"])
    iteration = 0
    best_score = 0.0
    scheduler = OneCycleLR(
        optimizer,
        total_steps=train_cfg["total_steps"],
        max_lr=train_cfg["lr"],
        pct_start=train_cfg["pct_start"],
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
    print(f"iteration: {iteration}")
    print(f"best_score: {best_score}")
    to_device = ToDevice(model_cfg["device"])
    scaler = GradScaler(enabled=use_amp)
    for _ in range((train_cfg["total_steps"] - iteration) // len(train_loader)):
        train_meter = MeanReduceDict()
        for batch, _ in tqdm(train_loader, total=len(train_loader)):
            iteration += 1
            model.train()
            loss_fn.train()
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
            scheduler.step(iteration)

            train_meter.update({"loss": loss.item()})

            if iteration % eval_interval == 0:
                model.eval()
                loss_fn.eval()
                metric = MeanAveragePrecisionK()
                val_meter = MeanReduceDict()
                with torch.no_grad():
                    for batch, annots in tqdm(val_loader, total=len(val_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)
                        logit = loss_fn.get_logits(embeddings)
                        pred_label_batch = logit.topk(k=5, dim=1)[1]
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
                score, _ = metric.value
                for k, v in train_meter.value.items():
                    writer.add_scalar(f"train/{k}", v, iteration)

                for k, v in val_meter.value.items():
                    writer.add_scalar(f"val/{k}", v, iteration)
                writer.add_scalar(f"val/loss", val_meter.value["loss"], iteration)
                writer.add_scalar(f"val/score", score, iteration)
                writer.add_scalar(f"train/lr", scheduler.get_last_lr()[0], iteration)
                train_meter.reset()
                metric.reset()
                val_meter.reset()

                if score > best_score:
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
                        "score": score,
                        "loss": loss,
                        "best_score": best_score,
                    },
                    target="latest",
                )
        writer.flush()


@torch.no_grad()
def evaluate(
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    fold: int,
    train_annotations: list[Annotation],
    val_annotations: list[Annotation],
    image_dir: str,
) -> list[Submission]:
    seed_everything()
    device = model_cfg["device"]
    writer = get_writer({**train_cfg, **model_cfg, **dataset_cfg})
    model = get_model(model_cfg).to(device)
    checkpoint = get_checkpoint(model_cfg)
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
        rows=train_annotations,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_annotations,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
    )

    reg_loader = DataLoader(
        reg_dataset,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
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
    to_device = ToDevice(model_cfg["device"])
    model.eval()
    val_meter = MeanReduceDict()
    metric = MeanAveragePrecisionK()
    matcher = MeanEmbeddingMatcher()

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
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    train_annotations: list[Annotation],
    test_annotations: list[Annotation],
    search_thresholds: list[dict],
    train_image_dir: str,
    test_image_dir: str,
) -> list[Submission]:
    seed_everything()
    device = model_cfg["device"]
    writer = get_writer({**train_cfg, **model_cfg, **dataset_cfg})
    model = get_model(model_cfg).to(device)
    checkpoint = get_checkpoint(model_cfg)
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
        rows=train_annotations,
        image_dir=train_image_dir,
        transform=Transform(dataset_cfg),
    )
    test_dataset = HwadCropedDataset(
        rows=test_annotations,
        image_dir=test_image_dir,
        transform=Transform(dataset_cfg),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    to_device = ToDevice(model_cfg["device"])
    threshold = topk(1, search_thresholds, key=lambda x: x["score"])[0]["threshold"]
    print(f"Threshold: {threshold}")
    model.eval()
    matcher = MeanEmbeddingMatcher()
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
    rows = add_new_individual(rows, threshold)
    return rows
