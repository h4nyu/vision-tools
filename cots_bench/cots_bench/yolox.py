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
import torch.nn.functional as F
from torch.optim import Adam
from vision_tools.utils import Checkpoint, load_config, batch_draw, merge_batch, draw
from torchvision.transforms.functional import vflip, hflip
from torch.utils.tensorboard import SummaryWriter
import functools
from datetime import datetime
from vision_tools.meter import MeanReduceDict
from vision_tools.box import box_hflip, box_vflip, resize_boxes
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch, TrainSample
from vision_tools.batch_transform import BatchMosaic
from vision_tools.metric import BoxAP
from torchvision.ops import nms
from cots_bench.data import (
    COTSDataset,
    TrainTransform,
    Transform,
    read_train_rows,
    collate_fn,
    kfold,
    filter_empty_boxes,
    keep_ratio,
)
from cots_bench.metric import BoxF2
from cots_bench.transform import RandomCutAndPaste, Resize
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
            cfg["mosaic_p"],
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


def get_tta_inference_one(cfg: Dict[str, Any]) -> "TTAInferenceOne":
    model = get_model(cfg)
    checkpoint = get_checkpoint(cfg)
    checkpoint.load_if_exists(
        model=model,
        device=cfg["device"],
        target=cfg["resume_target"],
    )
    to_boxes = get_to_boxes(cfg)
    resize = Resize(width=cfg["original_width"], height=cfg["original_height"])

    return TTAInferenceOne(
        model=model,
        to_boxes=to_boxes,
        postprocess=resize,
    )


def train(cfg: Dict[str, Any]) -> None:
    seed_everything()
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
    annotations = filter_empty_boxes(annotations)
    train_rows, validation_rows = kfold(annotations, cfg["n_splits"], cfg["fold"])
    train_non_zero_rows = pipe(
        train_rows, filter(lambda x: x["boxes"].shape[0] > 0), list
    )
    # train_zero_rows = pipe(train_rows, filter(lambda x: x["boxes"].shape[0] == 0), list)

    train_dataset = COTSDataset(
        train_non_zero_rows,
        transform=TrainTransform(cfg),
        random_cut_and_paste=RandomCutAndPaste(
            use_hflip=True,
            use_vflip=True,
            use_rot90=True,
            scale_limit=(0.9, 1.1),
            p=0.9,
        ),
    )

    # zero_dataset = COTSDataset(
    #     train_zero_rows,
    #     transform=TrainTransform(cfg),
    # )

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
        batch_size=cfg["train_loader"]["batch_size"],
        num_workers=cfg["train_loader"]["num_workers"],
    )
    # zero_loader = DataLoader(
    #     zero_dataset,
    #     sampler=RandomSampler(zero_dataset),
    #     batch_size=1,
    #     drop_last=True,
    #     num_workers=0,
    #     collate_fn=collate_fn,
    # )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg["val_loader"],
    )
    to_device = ToDevice(cfg["device"])
    mosaic = BatchMosaic(p=cfg["mosaic_p"])
    to_boxes = ToBoxes(
        iou_threshold=cfg["to_boxes"]["iou_threshold"],
        conf_threshold=0.01,
        limit=cfg["to_boxes"]["limit"],
    )
    iteration = 0
    use_amp = cfg["use_amp"]
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        meter = MeanReduceDict()
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = to_device(**batch)
            batch = mosaic(batch)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss, _, other = criterion(model, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            meter.accumulate(valmap(lambda x: x.item(), other))

        for k, v in meter.value.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        model.eval()
        meter = MeanReduceDict()
        ap = BoxAP()
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                batch = to_device(**batch)
                _, pred_yolo_batch, other = criterion(model, batch)
                pred_batch = to_boxes(pred_yolo_batch)
                ap.accumulate(
                    pred_batch["box_batch"],
                    pred_batch["conf_batch"],
                    batch["box_batch"],
                )
                meter.accumulate(valmap(lambda x: x.item(), other))
            ap_score, _ = ap.value
            writer.add_scalar(f"val/ap", ap_score, epoch)
            print(ap_score)
            print(meter.value)
            for k, v in meter.value.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            checkpoint.save_if_needed(model, ap_score, optimizer=optimizer)
        writer.flush()


@torch.no_grad()
def evaluate(cfg: Dict[str, Any]) -> None:
    seed_everything()
    checkpoint = get_checkpoint(cfg)
    writer = get_writer(cfg)
    model = get_model(cfg)
    checkpoint.load_if_exists(model=model, device=cfg["device"])
    annotations = read_train_rows(cfg["dataset_dir"])
    _, validation_rows = kfold(annotations, cfg["n_splits"], cfg["fold"])
    validation_rows = keep_ratio(validation_rows)
    writer = get_writer(cfg)
    dataset = COTSDataset(
        validation_rows[:100],
        transform=Transform(cfg),
    )
    to_device = ToDevice(cfg["device"])
    to_boxes = get_to_boxes(cfg)

    inference = TTAInferenceOne(
        model=model,
        to_boxes=to_boxes,
    )

    metric = BoxF2()
    ap = BoxAP()
    for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
        sample = to_device(**sample)
        pred_sample = inference(sample["image"])
        metric.accumulate([pred_sample["boxes"]], [sample["boxes"]])
        ap.accumulate([pred_sample["boxes"]], [pred_sample["confs"]], [sample["boxes"]])
        if i % 5 == 0 and pred_sample["boxes"].shape[0] > 0:
            plot = draw(
                image=pred_sample["image"],
                boxes=pred_sample["boxes"],
                gt_boxes=sample["boxes"],
            )
            writer.add_image("preview", plot, i // 5)
    score, other = metric.value
    ap_score, _ = ap.value
    print(f"f2:{score}, ap:{ap_score}, {other}")


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
        to_boxes: ToBoxes,
        postprocess: Optional[Resize] = None,
    ) -> None:
        self.model = model
        self.postprocess = postprocess
        self.to_boxes = to_boxes

    @torch.no_grad()
    def __call__(self, image: Tensor) -> TrainSample:
        self.model.eval()
        device = image.device
        _, h, w = image.shape
        vf_image = vflip(image)
        hf_image = hflip(image)
        vhf_image = hflip(vflip(image))
        image_batch = torch.stack(
            [
                image,
                vf_image,
                hf_image,
                vhf_image,
            ]
        )
        pred_yolo_batch = self.model(image_batch)
        pred_batch = self.to_boxes(pred_yolo_batch)
        boxes = pred_batch["box_batch"][0]
        vf_boxes = box_vflip(pred_batch["box_batch"][1], image_size=(w, h))
        hf_boxes = box_hflip(pred_batch["box_batch"][2], image_size=(w, h))
        vhf_boxes = box_vflip(
            box_hflip(pred_batch["box_batch"][3], image_size=(w, h)), image_size=(w, h)
        )
        weights = [1, 1, 1, 1]
        np_boxes, np_confs, np_lables = weighted_boxes_fusion(
            [
                resize_boxes(boxes, (1 / w, 1 / h)),
                resize_boxes(vf_boxes, (1 / w, 1 / h)),
                resize_boxes(hf_boxes, (1 / w, 1 / h)),
                resize_boxes(vhf_boxes, (1 / w, 1 / h)),
            ],
            pred_batch["conf_batch"],
            pred_batch["label_batch"],
            weights=weights,
        )
        sample = TrainSample(
            image=image_batch[0],
            boxes=resize_boxes(torch.from_numpy(np_boxes), (w, h)).to(device),
            labels=torch.from_numpy(np_lables).to(device),
            confs=torch.from_numpy(np_confs).to(device),
        )
        if self.postprocess is not None:
            sample = self.postprocess(sample)
        nms_idx = nms(sample["boxes"], sample["confs"], 0.5)
        sample["boxes"] = sample["boxes"][nms_idx]
        sample["labels"] = sample["labels"][nms_idx]
        sample["confs"] = sample["confs"][nms_idx]
        return sample


class EnsembleInferenceOne:
    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.model_configs = [load_config(p) for p in cfg["model_configs"]]
        presets = []
        for c in self.model_configs:
            model = get_model(c)
            checkpoint = get_checkpoint(c)
            checkpoint.load_if_exists(
                model=model,
                device=c["device"],
                target=c["resume_target"],
            )
            presets.append(
                {
                    "model": model,
                    "to_boxes": get_to_boxes(c),
                }
            )
        self.presets = presets
        self.resize = Resize(width=cfg["original_width"], height=cfg["original_height"])

    @torch.no_grad()
    def __call__(self, image: Tensor) -> TrainSample:
        device = image.device
        _, h, w = image.shape
        vf_image = vflip(image)
        hf_image = hflip(image)
        vhf_image = hflip(vflip(image))
        image_batch = torch.stack(
            [
                image,
                vf_image,
                hf_image,
                vhf_image,
            ]
        )
        pred_box_batch = []
        pred_conf_batch = []
        pred_label_batch = []

        for preset in self.presets:
            model = preset["model"].to(device)
            to_boxes = preset["to_boxes"]
            model.eval()
            pred_yolo_batch = model(image_batch)
            pred_batch = to_boxes(pred_yolo_batch)
            boxes = pred_batch["box_batch"][0]
            vf_boxes = box_vflip(pred_batch["box_batch"][1], image_size=(w, h))
            hf_boxes = box_hflip(pred_batch["box_batch"][2], image_size=(w, h))
            vhf_boxes = box_vflip(
                box_hflip(pred_batch["box_batch"][3], image_size=(w, h)),
                image_size=(w, h),
            )
            pred_box_batch += [
                resize_boxes(x, (1 / w, 1 / h))
                for x in [boxes, vf_boxes, hf_boxes, vhf_boxes]
            ]
            pred_conf_batch += pred_batch["conf_batch"]
            pred_label_batch += pred_batch["label_batch"]

        np_boxes, np_confs, np_lables = weighted_boxes_fusion(
            pred_box_batch,
            pred_conf_batch,
            pred_label_batch,
        )

        sample = TrainSample(
            image=image_batch[0],
            boxes=resize_boxes(torch.from_numpy(np_boxes), (w, h)).to(device),
            labels=torch.from_numpy(np_lables).to(device),
            confs=torch.from_numpy(np_confs).to(device),
        )
        sample = self.resize(sample)
        nms_idx = nms(sample["boxes"], sample["confs"], 0.5)
        sample["boxes"] = sample["boxes"][nms_idx]
        sample["labels"] = sample["labels"][nms_idx]
        sample["confs"] = sample["confs"][nms_idx]
        return sample
