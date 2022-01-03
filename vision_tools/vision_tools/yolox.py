import torch
from torch import nn
from torch import Tensor
from typing import Callable, TypedDict, Any
from torch.cuda.amp import GradScaler, autocast
import math
from torchvision.ops import box_convert, batched_nms
import torch.nn.functional as F

from .block import DefaultActivation, DWConv, ConvBnAct
from .interface import FPNLike, BackboneLike, TrainBatch
from .assign import SimOTA
from .loss import CIoULoss, FocalLoss, DIoULoss
from toolz import valmap


class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_channels: int,
        act: Callable = DefaultActivation,
        depthwise: bool = False,
    ):
        super().__init__()
        Conv = DWConv if depthwise else ConvBnAct
        self.stem = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act=act,
        )
        self.reg_branch = nn.Sequential(
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )

        self.cls_branch = nn.Sequential(
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.cls_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.init_weights()

    def init_weights(self, prior_prob: float = 1e-2) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        for conv in [self.cls_out, self.obj_out]:
            if conv.bias is None:
                continue
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        reg_feat = self.reg_branch(stem)
        cls_feat = self.cls_branch(stem)
        reg_out = self.reg_out(reg_feat)
        obj_out = self.obj_out(reg_feat)
        cls_out = self.cls_out(cls_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class YOLOXHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hidden_channels: int,
        num_classes: int = 1,
        depthwise: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                DecoupledHead(
                    in_channels=c,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    act=act,
                )
                for c in in_channels
            ]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        return [m(x) for m, x in zip(self.heads, feats)]


class YOLOX(nn.Module):
    def __init__(
        self,
        backbone: BackboneLike,
        neck: FPNLike,
        hidden_channels: int,
        num_classes: int,
        box_limit: int = 300,
        box_iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        feat_range: tuple[int, int] = (3, 7),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.feat_range = feat_range
        self.num_classes = num_classes
        self.box_limit = box_limit
        self.strides = self.backbone.strides
        self.box_strides = self.strides[self.feat_range[0] : self.feat_range[1]]
        self.box_iou_threshold = box_iou_threshold
        self.score_threshold = score_threshold

        self.box_head = YOLOXHead(
            in_channels=backbone.channels[self.feat_range[0] : self.feat_range[1]],
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )

    def box_branch(self, feats: list[Tensor]) -> Tensor:
        device = feats[0].device
        box_levels = self.box_head(feats)
        yolo_box_list = []
        for pred, stride in zip(box_levels, self.box_strides):
            batch_size, num_outputs, rows, cols = pred.shape
            grid = (
                torch.stack(
                    torch.meshgrid([torch.arange(rows), torch.arange(cols)])[::-1],
                    dim=2,
                )
                .reshape(rows * cols, 2)
                .to(device)
            )
            strides = torch.full((batch_size, len(grid), 1), stride).to(device)
            anchor_points = (grid + 0.5) * strides
            yolo_boxes = (
                pred.permute(0, 2, 3, 1)
                .reshape(batch_size, rows * cols, num_outputs)
                .float()
            )
            yolo_boxes = torch.cat(
                [
                    (yolo_boxes[..., 0:2] + 0.5 + grid) * strides,
                    yolo_boxes[..., 2:4] ** 2 * strides,
                    yolo_boxes[..., 4:],
                    strides,
                    anchor_points,
                ],
                dim=-1,
            )
            yolo_box_list.append(yolo_boxes)
        yolo_batch = torch.cat(yolo_box_list, dim=1)
        return yolo_batch

    def feats(self, x: Tensor) -> list[Tensor]:
        feats = self.backbone(x)
        feats = feats[self.feat_range[0] : self.feat_range[1]]
        return self.neck(feats)

    @torch.no_grad()
    def to_boxes(
        self, yolo_batch: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        score_batch, box_batch, lable_batch = [], [], []
        device = yolo_batch.device
        num_classes = self.num_classes
        for r in yolo_batch:
            scores = r[:, 4].sigmoid()
            th_filter = scores > self.score_threshold
            r = r[th_filter]
            scores = scores[th_filter]
            boxes = box_convert(
                r[:, :4], in_fmt="cxcywh", out_fmt="xyxy"
            )
            lables = r[:, 5 : 5 + num_classes].argmax(-1).long()
            nms_index = batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=lables,
                iou_threshold=self.box_iou_threshold,
            )[: self.box_limit]
            box_batch.append(boxes[nms_index])
            score_batch.append(scores[nms_index])
            lable_batch.append(lables[nms_index])
        return score_batch, box_batch, lable_batch

    def forward(self, image_batch: Tensor) -> TrainBatch:
        feats = self.feats(image_batch)
        yolo_batch = self.box_branch(feats)
        score_batch, box_batch, label_batch = self.to_boxes(yolo_batch)
        return dict(
            image_batch=image_batch,
            box_batch=box_batch,
            label_batch=label_batch,
        )


class Criterion:
    def __init__(
        self,
        assign: SimOTA,
        obj_weight: float = 1.0,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
        assign_radius: float = 2.0,
        assign_center_wight: float = 1.0,
    ) -> None:
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.assign = assign

        self.box_loss = DIoULoss()
        # self.box_loss = nn.L1Loss()
        self.obj_loss = F.binary_cross_entropy_with_logits
        self.cls_loss = F.binary_cross_entropy_with_logits

    def __call__(
        self,
        model: YOLOX,
        inputs: TrainBatch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images = inputs["image_batch"]
        gt_box_batch = inputs["box_batch"]
        gt_label_batch = inputs["label_batch"]
        device = images.device
        num_classes = model.num_classes
        feats = model.feats(images)
        pred_yolo_batch = model.box_branch(feats)
        gt_yolo_batch, pos_idx = self.prepeare_box_gt(
            model.num_classes, gt_box_batch, gt_label_batch, pred_yolo_batch
        )

        # 1-stage
        obj_loss = self.obj_loss(pred_yolo_batch[..., 4], gt_yolo_batch[..., 4])

        box_loss, cls_loss = (
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        matched_count = pos_idx.sum()
        if matched_count > 0:
            box_loss += self.box_loss(
                box_convert(
                    pred_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                box_convert(
                    gt_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
                ),
            )
            if self.cls_weight != 0:
                cls_loss += self.cls_loss(
                    pred_yolo_batch[..., 5 : 5 + num_classes][pos_idx],
                    gt_yolo_batch[..., 5 : 5 + num_classes][pos_idx],
                )

        loss = (
            self.box_weight * box_loss
            + self.obj_weight * obj_loss
            + self.cls_weight * cls_loss
        )
        return (
            loss,
            dict(loss=loss, obj_loss=obj_loss, box_loss=box_loss, cls_loss=cls_loss),
        )

    @torch.no_grad()
    def prepeare_box_gt(
        self,
        num_classes: int,
        gt_boxes_batch: list[Tensor],
        gt_label_batch: list[Tensor],
        pred_yolo_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = pred_yolo_batch.device
        gt_yolo_batch = torch.zeros(
            pred_yolo_batch.shape,
            dtype=pred_yolo_batch.dtype,
            device=device,
        )

        for batch_idx, (gt_boxes, gt_labels, pred_yolo) in enumerate(
            zip(gt_boxes_batch, gt_label_batch, pred_yolo_batch)
        ):
            gt_cxcywh = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")
            matched = self.assign(
                gt_boxes=gt_boxes,
                pred_boxes=box_convert(
                    pred_yolo[..., :4], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                pred_scores=pred_yolo[..., 4].sigmoid(),
                strides=pred_yolo[..., 5 + num_classes],
                anchor_points=pred_yolo[..., 6 + num_classes : 6 + num_classes + 2],
            )
            gt_yolo_batch[batch_idx, matched[:, 1], :4] = gt_cxcywh[matched[:, 0]]
            gt_yolo_batch[batch_idx, matched[:, 1], 4] = 1.0
            gt_yolo_batch[batch_idx, matched[:, 1], 5 : 5 + num_classes] = F.one_hot(
                gt_labels[matched[:, 0]], num_classes
            ).to(gt_yolo_batch)

        pos_idx = gt_yolo_batch[..., 4] == 1.0
        return gt_yolo_batch, pos_idx


class Inference:
    def __call__(
        self,
        model: YOLOX,
        inputs: TrainBatch,
    ) -> TrainBatch:
        return model(inputs["image_batch"])
