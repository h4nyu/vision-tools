import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from toolz import valmap
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import batched_nms, box_convert
from typing_extensions import TypedDict

from .anchors import Anchor
from .assign import SimOTA
from .block import ConvBnAct, DefaultActivation, DWConv
from .interface import BackboneLike, FPNLike, TrainBatch
from .loss import CIoULoss, DIoULoss, FocalLossWithLogits


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
        in_channels: List[int],
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

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        return [m(x) for m, x in zip(self.heads, feats)]


class YOLOX(nn.Module):
    def __init__(
        self,
        backbone: BackboneLike,
        neck: FPNLike,
        hidden_channels: int,
        num_classes: int,
        feat_range: Tuple[int, int] = (3, 7),
        head_range: Tuple[int, int] = (0, 3),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.feat_range = feat_range
        self.head_range = head_range
        self.num_classes = num_classes
        self.strides = self.backbone.strides
        self.box_strides = self.neck.strides[self.head_range[0] : self.head_range[1]]

        self.box_head = YOLOXHead(
            in_channels=self.neck.channels[self.head_range[0] : self.head_range[1]],
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )
        self.anchor = Anchor()

    def box_branch(self, feats: List[Tensor]) -> Tensor:
        device = feats[0].device
        box_levels = self.box_head(feats)
        yolo_box_List = []
        for pred, stride in zip(box_levels, self.box_strides):
            batch_size, num_outputs, height, width = pred.shape
            anchor_boxes = self.anchor(
                height=height, width=width, stride=stride, device=device
            )
            anchor_points = (anchor_boxes[:, 0:2] + anchor_boxes[:, 2:4]) / 2.0
            yolo_boxes = (
                pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_outputs)
            )
            strides = torch.full((batch_size, yolo_boxes.size(1), 1), stride).to(device)
            yolo_boxes = torch.cat(
                [
                    yolo_boxes[..., 0:2] * strides + anchor_points,
                    (yolo_boxes[..., 2:4].exp()) * strides,
                    yolo_boxes[..., 4:],
                    strides,
                    anchor_points.unsqueeze(0).expand(batch_size, *anchor_points.shape),
                ],
                dim=-1,
            )
            yolo_box_List.append(yolo_boxes)
        yolo_batch = torch.cat(yolo_box_List, dim=1)
        return yolo_batch

    def feats(self, x: Tensor) -> List[Tensor]:
        feats = self.backbone(x)
        feats = feats[self.feat_range[0] : self.feat_range[1]]
        return self.neck(feats)[self.head_range[0] : self.head_range[1]]

    def forward(self, image_batch: Tensor) -> Tensor:
        feats = self.feats(image_batch)
        return self.box_branch(feats)


class ToBoxes:
    def __init__(
        self,
        limit: int = 300,
        iou_threshold: Optional[float] = None,
        conf_threshold: float = 0.5,
    ):
        self.limit = limit
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    @torch.no_grad()
    def __call__(self, yolo_batch: Tensor) -> Dict[str, List[Tensor]]:
        conf_batch, box_batch, label_batch = [], [], []
        device = yolo_batch.device
        num_classes = yolo_batch.shape[-1] - 5
        for r in yolo_batch:
            scores = r[:, 4].sigmoid()
            th_filter = scores > self.conf_threshold
            scores, sort_idx = torch.sort(scores[th_filter], descending=True)
            sort_idx = sort_idx[: self.limit]
            r = r[th_filter][sort_idx]
            boxes = box_convert(r[:, :4], in_fmt="cxcywh", out_fmt="xyxy")
            lables = r[:, 5 : 5 + num_classes].argmax(-1).long()
            if self.iou_threshold is not None:
                nms_index = batched_nms(
                    boxes=boxes,
                    scores=scores,
                    idxs=lables,
                    iou_threshold=self.iou_threshold,
                )
                boxes = boxes[nms_index]
                scores = scores[nms_index]
                lables = lables[nms_index]
            box_batch.append(boxes)
            conf_batch.append(scores)
            label_batch.append(lables)
        return dict(box_batch=box_batch, conf_batch=conf_batch, label_batch=label_batch)


class Criterion:
    def __init__(
        self,
        assign: SimOTA,
        obj_weight: float = 1.0,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
    ) -> None:
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.assign = assign

        self.box_loss = CIoULoss()
        self.obj_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.cls_loss = F.binary_cross_entropy_with_logits

    def __call__(
        self,
        model: YOLOX,
        inputs: TrainBatch,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
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
        matched_count = pos_idx.sum()

        # 1-stage
        obj_loss = self.obj_loss(
            pred_yolo_batch[..., 4], gt_yolo_batch[..., 4]
        ) / matched_count.clamp(min=1)
        box_loss, cls_loss = (
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
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
            pred_yolo_batch,
            dict(loss=loss, obj_loss=obj_loss, box_loss=box_loss, cls_loss=cls_loss),
        )

    @torch.no_grad()
    def prepeare_box_gt(
        self,
        num_classes: int,
        gt_box_batch: List[Tensor],
        gt_label_batch: List[Tensor],
        pred_yolo_batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        device = pred_yolo_batch.device
        gt_yolo_batch = torch.zeros(
            (pred_yolo_batch.size(0), pred_yolo_batch.size(1), 5 + num_classes),
            dtype=pred_yolo_batch.dtype,
            device=device,
        )

        for batch_idx, (gt_boxes, gt_labels, pred_yolo) in enumerate(
            zip(gt_box_batch, gt_label_batch, pred_yolo_batch)
        ):
            if len(gt_boxes) == 0:
                continue
            matched = self.assign(
                gt_boxes=gt_boxes,
                pred_boxes=box_convert(
                    pred_yolo[:, :4], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                pred_objs=pred_yolo[:, 4],
                strides=pred_yolo[:, 5 + num_classes],
                anchor_points=pred_yolo[:, 6 + num_classes : 6 + num_classes + 2],
            )
            gt_yolo_batch[batch_idx, matched[:, 1], :4] = box_convert(
                gt_boxes, in_fmt="xyxy", out_fmt="cxcywh"
            )[matched[:, 0]]
            gt_yolo_batch[batch_idx, matched[:, 1], 4] = 1.0
            gt_yolo_batch[batch_idx, matched[:, 1], 5 : 5 + num_classes] = (
                F.one_hot(gt_labels[matched[:, 0]], num_classes).float().to(device)
            )

        pos_idx = gt_yolo_batch[..., 4] == 1.0
        return gt_yolo_batch, pos_idx


class Inference:
    def __call__(
        self,
        model: YOLOX,
        inputs: TrainBatch,
    ) -> TrainBatch:
        return model(inputs["image_batch"])
