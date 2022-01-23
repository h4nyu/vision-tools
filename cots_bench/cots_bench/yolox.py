from typing import Any, Dict, Optional
import torch
import os
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset, DataLoader, RandomSampler
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, ToBoxes
from vision_tools.backbone import CSPDarknet, EfficientNet
from vision_tools.neck import CSPPAFPN
from vision_tools.yolox import YOLOX, Criterion
from vision_tools.assign import SimOTA
from vision_tools.optim import Lookahead
from torch.optim import Adam
from vision_tools.utils import Checkpoint, load_config, batch_draw, merge_batch
from torchvision.transforms.functional import vflip, hflip
from torch.utils.tensorboard import SummaryWriter
import functools
from datetime import datetime
from vision_tools.meter import MeanReduceDict
from vision_tools.box import box_hflip, box_vflip, resize_boxes
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch, TrainSample
from vision_tools.batch_transform import BatchMosaic
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    InferenceTransform,
    read_train_rows,
    collate_fn,
    kfold,
    filter_empty_boxes,
    keep_ratio,
)
from cots_bench.metric import BoxF2
from cots_bench.transform import RandomCutAndPaste
from toolz.curried import pipe, partition, map, filter, count, valmap


def get_model_name(cfg: Dict[str, Any]) -> str:
    return pipe(
        [
            cfg["name"],
            cfg["backbone_name"],
            cfg["hidden_channels"],
            cfg["fold"],
            cfg["n_splits"],
            cfg["image_width"],
            cfg["image_height"],
            cfg["fpn_start"],
            cfg["fpn_end"],
        ],
        map(str),
        "-".join,
    )


def get_writer(cfg: Dict[str, Any]) -> SummaryWriter:
    model_name = get_model_name(cfg)
    writer_name = pipe(
        [
            model_name,
            cfg["lr"],
            cfg["criterion"]["box_weight"],
            cfg["assign"]["radius"],
            "mosaic",
            "cutout",
            "scale-0.5-1.0",
            "cut_and_paste",
            "roteate90",
        ],
        map(str),
        "-".join,
    )
    return SummaryWriter(
        f"runs/{writer_name}",
    )


def get_model(cfg: Dict[str, Any]) -> YOLOX:
    backbone = EfficientNet(name=cfg["backbone_name"])
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg["fpn_start"] : cfg["fpn_end"]],
        strides=backbone.strides[cfg["fpn_start"] : cfg["fpn_end"]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg["hidden_channels"],
        num_classes=cfg["num_classes"],
        feat_range=(cfg["fpn_start"], cfg["fpn_end"]),
    )
    return model


def get_to_boxes(cfg: Dict[str, Any]) -> ToBoxes:
    return ToBoxes(**cfg["to_boxes"])


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


def get_tta_inference_one(cfg: Dict[str, Any]) -> "TTAInferenceOne":
    model = get_model(cfg)
    checkpoint = get_checkpoint(cfg)
    checkpoint.load_if_exists(
        model=model,
        device=cfg["device"],
        target=cfg["resume_target"],
    )

    return TTAInferenceOne(
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
    optimizer = Adam(
        model.parameters(),
        lr=cfg["lr"],
    )
    checkpoint.load_if_exists(
        model=model,
        optimizer=optimizer,
        device=cfg["device"],
        target=cfg["resume_target"],
    )

    annotations = read_train_rows(cfg["dataset_dir"])
    train_rows, validation_rows = kfold(annotations, cfg["n_splits"], cfg["fold"])
    train_non_zero_rows = pipe(
        train_rows, filter(lambda x: x["boxes"].shape[0] > 0), list
    )
    train_zero_rows = pipe(train_rows, filter(lambda x: x["boxes"].shape[0] == 0), list)

    train_dataset = COTSDataset(
        train_non_zero_rows,
        transform=TrainTransform(cfg),
        random_cut_and_paste=RandomCutAndPaste(use_hflip=True, use_vflip=True, use_rot90=True),
    )

    zero_dataset = COTSDataset(
        train_zero_rows,
        transform=TrainTransform(cfg),
    )

    print(f"train_dataset={train_dataset}")
    val_dataset = COTSDataset(
        keep_ratio(validation_rows),
        transform=Transform(cfg),
    )
    print(f"val_dataset={val_dataset}")
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        batch_size=cfg["train_loader"]["batch_size"] - 1,
        num_workers=cfg["train_loader"]["num_workers"],
    )
    zero_loader = DataLoader(
        zero_dataset,
        sampler=RandomSampler(zero_dataset),
        batch_size=1,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg["val_loader"],
    )
    to_device = ToDevice(cfg["device"])
    metric = BoxF2(iou_thresholds=cfg["val_iou_thresholds"])
    mosaic = BatchMosaic(p=0.3)
    to_boxes = get_to_boxes(cfg)
    iteration = 0
    use_amp = cfg["use_amp"]
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        meter = MeanReduceDict()
        for (batch, zero_batch) in tqdm(
            zip(train_loader, zero_loader), total=len(train_loader)
        ):
            batch = to_device(**batch)
            zero_batch = to_device(**zero_batch)
            merged_batch = merge_batch([batch, zero_batch])
            merged_batch = mosaic(merged_batch)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss, _, other = criterion(model, merged_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            meter.accumulate(valmap(lambda x: x.item(), other))
        for k, v in meter.value.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        model.eval()
        meter = MeanReduceDict()
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                batch = to_device(**batch)
                _, pred_yolo_batch, other = criterion(model, batch)
                pred_box_batch = to_boxes(pred_yolo_batch)["box_batch"]
                metric.accumulate(pred_box_batch, batch["box_batch"])
                meter.accumulate(valmap(lambda x: x.item(), other))
            score, other_log = metric.value
            writer.add_scalar(f"val/score", score, epoch)
            for k, v in other_log.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(metric.value)
            print(meter.value)
            for k, v in meter.value.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            checkpoint.save_if_needed(model, score, optimizer=optimizer)
        writer.flush()


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
        transform=Transform(cfg),
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


class TTAInferenceOne:
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
        _, h, w = image.shape
        vf_image = vflip(image)
        hf_image = hflip(image)
        image_batch = self.to_device(
            image_batch=torch.stack(
                [
                    image,
                    vf_image,
                    hf_image,
                ]
            )
        )["image_batch"]
        pred_batch = self.model(image_batch)
        boxes = pred_batch["box_batch"][0]
        vf_boxes = box_vflip(pred_batch["box_batch"][1], image_size=(w, h))
        hf_boxes = box_hflip(pred_batch["box_batch"][2], image_size=(w, h))
        np_boxes, np_confs, np_lables = weighted_boxes_fusion(
            [
                resize_boxes(boxes, (1 / w, 1 / h)),
                resize_boxes(vf_boxes, (1 / w, 1 / h)),
                resize_boxes(hf_boxes, (1 / w, 1 / h)),
            ],
            pred_batch["conf_batch"],
            pred_batch["label_batch"],
        )
        if self.postprocess is not None:
            pred_batch = self.postprocess(pred_batch)
        return TrainSample(
            image=pred_batch["image_batch"][0],
            boxes=resize_boxes(torch.from_numpy(np_boxes), (w, h)),
            labels=torch.from_numpy(np_lables),
            confs=torch.from_numpy(np_confs),
        )
