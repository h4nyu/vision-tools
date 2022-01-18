from typing import Any, Dict, Optional
import torch
import os
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, Inference
from vision_tools.backbone import CSPDarknet, EfficientNet
from vision_tools.neck import CSPPAFPN
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.assign import SimOTA
from vision_tools.utils import Checkpoint, load_config, batch_draw
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from vision_tools.meter import MeanReduceDict
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch, TrainSample
from vision_tools.batch_transform import BatchRemovePadding
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    InferenceTransform,
    read_train_rows,
    collate_fn,
    kfold,
    filter_empty_boxes,
)
from cots_bench.metric import BoxF2
from toolz.curried import pipe, filter


def get_model_name(cfg: Dict[str, Any]) -> str:
    return f"{cfg['name']}-{cfg['fold']}-{cfg['feat_range'][0]}-{cfg['feat_range'][1]}-{cfg['hidden_channels']}-{cfg['backbone_name']}"


def get_writer(cfg: Dict[str, Any]) -> SummaryWriter:
    model_name = get_model_name(cfg)
    return SummaryWriter(
        f"runs/{model_name}-lr_{cfg['lr']}-box_w_{cfg['criterion']['box_weight']}-radius_{cfg['assign']['radius']}"
    )


def get_model(cfg: Dict[str, Any]) -> YOLOX:
    backbone = EfficientNet(name=cfg["backbone_name"])
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg["feat_range"][0] : cfg["feat_range"][1]],
        strides=backbone.strides[cfg["feat_range"][0] : cfg["feat_range"][1]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg["hidden_channels"],
        num_classes=cfg["num_classes"],
        feat_range=cfg["feat_range"],
        box_iou_threshold=cfg["box_iou_threshold"],
        score_threshold=cfg["score_threshold"],
    )
    return model


def get_criterion(cfg: Dict[str, Any]) -> Criterion:
    assign = SimOTA(**cfg["assign"])
    criterion = Criterion(assign=assign, **cfg["criterion"])
    return criterion


def get_checkpoint(cfg: Dict[str, Any]) -> Checkpoint:
    return Checkpoint[YOLOX](
        root_path=os.path.join(cfg["store_dir"], get_model_name(cfg)),
        default_score=0.0,
    )


def get_inference_one(cfg: Dict[str, Any]) -> "InferenceOne":
    model = get_model(cfg)
    checkpoint = get_checkpoint(cfg)
    checkpoint.load_if_exists(
        model=model,
        device=cfg["device"],
        target=cfg["resume_target"],
    )

    return InferenceOne(
        model=model,
        transform=InferenceTransform(),
        to_device=ToDevice(cfg["device"]),
    )


def train() -> None:
    seed_everything()
    cfg = load_config(os.path.join(os.path.dirname(__file__), "../config/yolox.yaml"))
    checkpoint = get_checkpoint(cfg)
    writer = get_writer(cfg)
    model = get_model(cfg)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    checkpoint.load_if_exists(
        model=model,
        optimizer=optimizer,
        device=cfg["device"],
        target=cfg["resume_target"],
    )

    annotations = read_train_rows(cfg["dataset_dir"])
    annotations = filter_empty_boxes(annotations)
    train_rows, validation_rows = kfold(annotations, cfg["n_splits"], cfg["fold"])
    print(len(train_rows), len(validation_rows))
    train_rows = pipe(train_rows, filter(lambda row: len(row["boxes"]) > 0), list)
    train_dataset = COTSDataset(
        train_rows,
        transform=TrainTransform(),
    )
    val_dataset = COTSDataset(
        validation_rows,
        transform=Transform(),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg["train_loader"],
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg["val_loader"],
    )
    to_device = ToDevice(cfg["device"])

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=to_device,
        criterion=criterion,
        optimizer=optimizer,
        loader=train_loader,
        meter=MeanReduceDict(),
        writer=writer,
        checkpoint=checkpoint,
        use_amp=cfg["use_amp"],
    )
    metric = BoxF2()

    eval_step = EvalStep[YOLOX, TrainBatch](
        to_device=to_device,
        loader=val_loader,
        metric=metric,
        writer=writer,
        inference=Inference(),
        checkpoint=checkpoint,
    )

    for epoch in range(cfg["num_epochs"]):
        train_step(model, epoch)
        eval_step(model, epoch)


@torch.no_grad()
def evaluate() -> None:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "../config/yolox.yaml"))
    checkpoint = get_checkpoint(cfg)
    writer = get_writer(cfg)
    model = get_model(cfg)
    checkpoint.load_if_exists(model=model, device=cfg["device"])
    annotations = read_train_rows(cfg["dataset_dir"])
    annotations = filter_empty_boxes(annotations)
    _, annotations = kfold(annotations, cfg["n_splits"], cfg["fold"])
    # train_rows = pipe(train_rows, filter(lambda row: len(row["boxes"]) > 0), list)
    dataset = COTSDataset(
        annotations,
        transform=Transform(),
    )
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **cfg["val_loader"],
    )
    to_device = ToDevice(cfg["device"])
    metric = BoxF2()
    model.eval()
    for i, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = to_device(**batch)
        pred_batch = model(batch["image_batch"])
        metric.accumulate(pred_batch, batch)
    score, other = metric.value
    print(score, other)
    writer.add_scalar(f"evaluate-all/score", score, 0)

class InferenceOne:
    def __init__(
        self,
        model: YOLOX,
        transform: Any,
        to_device: ToDevice,
        postprocess: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.to_device = to_device
        self.postprocess = postprocess

    @torch.no_grad()
    def __call__(self, image: Any) -> TrainSample:
        self.model.eval()
        transformed = self.transform(image=image)
        image = (transformed["image"] / 255).float()
        image_batch = self.to_device(image_batch=image.unsqueeze(dim=0))["image_batch"]
        pred_batch = self.model(image_batch)
        if self.postprocess is not None:
            pred_batch = self.postprocess(pred_batch)
        return TrainSample(
            image=pred_batch["image_batch"][0],
            boxes=pred_batch["box_batch"][0],
            labels=pred_batch["label_batch"][0],
            confs=pred_batch["conf_batch"][0],
        )
