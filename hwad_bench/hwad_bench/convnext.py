from __future__ import annotations

import timm
import torch
from pytorch_metric_learning.distances import CosineSimilarity
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
    Submission,
    TrainTransform,
    Transform,
    collate_fn,
    filter_annotations_by_fold,
    filter_in_annotations,
)
from hwad_bench.metrics import MeanAveragePrecisionK
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


class MeanEmbeddingMmatcher:
    def __init__(self) -> None:
        self.embeddings: dict[int, list[Tensor]] = {}
        self.index: Tensor = torch.zeros(0, 0)
        self.max_classes = 0
        self.embedding_size = 0
        self.distance = CosineSimilarity()

    def update(self, embeddings: Tensor, labels: Tensor) -> None:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        for label, embedding in zip(labels, embeddings):
            label = int(label)
            old_embedding = self.embeddings.get(label, [])
            self.embeddings[label] = old_embedding + [embedding]
            if self.max_classes < label:
                self.max_classes = label
            self.embedding_size = embedding.shape[-1]

    def create_index(self) -> Tensor:
        index = torch.full((self.max_classes + 1, self.embedding_size), float("nan"))
        if len(list(self.embeddings.values())) > 0:
            device = list(self.embeddings.values())[0][0]
            index = index.to(device)

        for label, embeddings in self.embeddings.items():
            index[int(label)] = torch.mean(torch.stack(embeddings), dim=0)
        self.index = index
        return index

    def __call__(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.index).nan_to_num(float("-inf"))


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
            cfg["random_resized_crop_p"],
            cfg["min_samples"],
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
    writer = get_writer({**train_cfg, **model_cfg, **dataset_cfg})

    train_annots = filter_annotations_by_fold(
        annotations, fold_train, min_samples=train_cfg["min_samples"]
    )

    # TODO num_class
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
    best_score = 0.0
    best_loss = float("inf")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        loss_fn.load_state_dict(saved_state["loss_fn"])
        optimizer.load_state_dict(saved_state["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        iteration = saved_state.get("iteration", 0)
        best_loss = saved_state.get("best_loss", float("inf"))
        best_score = saved_state.get("best_score", 0.0)

    reg_annots = filter_annotations_by_fold(annotations, fold_train, min_samples=0)

    val_annots = filter_annotations_by_fold(annotations, fold_val, min_samples=0)
    val_annots = filter_in_annotations(val_annots, reg_annots)
    train_dataset = HwadCropedDataset(
        rows=train_annots,
        image_dir=image_dir,
        transform=TrainTransform(dataset_cfg),
    )

    reg_dataset = HwadCropedDataset(
        rows=reg_annots,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
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

    reg_loader = DataLoader(
        reg_dataset,
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
    to_device = ToDevice(model_cfg["device"])
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(train_cfg["epochs"]):
        train_meter = MeanReduceDict()
        for batch in tqdm(train_loader, total=len(train_loader)):
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

            train_meter.update({"loss": loss.item()})

            if iteration % eval_interval == 0:
                model.eval()
                loss_fn.eval()
                val_meter = MeanReduceDict()
                metric = MeanAveragePrecisionK()
                matcher = MeanEmbeddingMmatcher()
                with torch.no_grad():
                    for batch in tqdm(reg_loader, total=len(reg_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)
                        matcher.update(embeddings, label_batch)
                    matcher.create_index()
                    for batch in tqdm(val_loader, total=len(val_loader)):
                        batch = to_device(**batch)
                        image_batch = batch["image_batch"]
                        label_batch = batch["label_batch"]
                        embeddings = model(image_batch)

                        loss = loss_fn(
                            embeddings,
                            label_batch,
                        )
                        distance = matcher(embeddings)
                        pred_label_batch = distance.topk(k=5, dim=1)[1]
                        val_meter.update({"loss": loss.item()})
                        metric.update(
                            pred_label_batch,
                            label_batch,
                        )

                score, _ = metric.value
                loss = val_meter.value["loss"]
                for k, v in train_meter.value.items():
                    writer.add_scalar(f"train/{k}", v, iteration)
                for k, v in val_meter.value.items():
                    writer.add_scalar(f"val/{k}", v, iteration)
                writer.add_scalar(f"val/score", score, iteration)

                if score > best_score:
                    best_score = score
                    checkpoint.save(
                        {
                            "model": model.state_dict(),
                            "loss_fn": loss_fn.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "score": score,
                            "loss": loss,
                            "best_score": best_score,
                            "best_loss": best_loss,
                        },
                        target="best_score",
                    )

                if loss < best_loss:
                    best_loss = loss
                    checkpoint.save(
                        {
                            "model": model.state_dict(),
                            "loss_fn": loss_fn.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "score": score,
                            "loss": loss,
                            "best_score": best_score,
                            "best_loss": best_loss,
                        },
                        target="best_loss",
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
                        "best_loss": best_loss,
                    },
                    target="latest",
                )
                train_meter.reset()
                val_meter.reset()
                metric.reset()

        writer.flush()


@torch.no_grad()
def evaluate(
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    fold: int,
    annotations: list[Annotation],
    fold_train: list[dict],
    fold_val: list[dict],
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

    reg_annots = filter_annotations_by_fold(annotations, fold_train, min_samples=0)
    val_annots = filter_annotations_by_fold(annotations, fold_val, min_samples=0)
    val_annots = filter_in_annotations(val_annots, reg_annots)

    reg_dataset = HwadCropedDataset(
        rows=reg_annots,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
    )
    val_dataset = HwadCropedDataset(
        rows=val_annots,
        image_dir=image_dir,
        transform=Transform(dataset_cfg),
    )

    reg_loader = DataLoader(
        reg_dataset,
        batch_size=train_cfg["batch_size"],
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
    matcher = MeanEmbeddingMmatcher()
    label_id_map = {}
    for batch, batch_annot in tqdm(reg_loader, total=len(reg_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        ids = pipe(batch_annot, map(lambda x: x["individual_id"]), list)
        for id, label in zip(ids, label_batch):
            label_id_map[int(label)] = id
        embeddings = model(image_batch)
        matcher.update(embeddings, label_batch)
    matcher.create_index()
    rows: list[Submission] = []
    for batch, batch_annots in tqdm(val_loader, total=len(val_loader)):
        batch = to_device(**batch)
        image_batch = batch["image_batch"]
        label_batch = batch["label_batch"]
        embeddings = model(image_batch)
        distance = matcher(embeddings)
        topk_distance, pred_label_batch = distance.topk(k=5, dim=1)
        for pred_topk, annot, distances in zip(
            pred_label_batch.tolist(), batch_annots, topk_distance.tolist()
        ):
            individual_ids = pipe(
                pred_topk,
                map(lambda x: label_id_map[x]),
                list,
            )
            row = Submission(
                image_file=annot["image_file"],
                distances=distances,
                individual_ids=individual_ids,
            )
            rows.append(row)
    return rows
