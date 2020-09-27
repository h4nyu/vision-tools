import torch
import typing
from torch import nn


FocsBoxes = typing.NewType("FocsBoxes", torch.Tensor)

LogitMap = typing.NewType("LogitMap", torch.Tensor)  # [B, N, H, W]
BoxMap = typing.NewType("BoxMap", torch.Tensor)  # [B, 4, H, W]
CenterMap = typing.NewType("CenterMap", torch.Tensor)  # [B, 1, H, W]
Features = typing.List[torch.Tensor]

BoxMaps = typing.List[BoxMap]
CenterMaps = typing.List[CenterMap]
LogitMaps = typing.List[LogitMap]

NetOut = typing.Tuple[LogitMaps, CenterMaps, BoxMaps]


def init_conv_kaiming(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module: nn.Module, std: float = 0.01) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def centerness(boxes: FocsBoxes) -> torch.Tensor:
    l, t, r, b = boxes.unbind(-1)
    return torch.sqrt(
        (torch.min(l, r) * torch.min(t, b))
        .true_divide(torch.max(l, r))
        .true_divide(torch.max(t, b))
    )


class Head(nn.Module):
    def __init__(
        self, depth: int, in_channels: int, n_classes: int
    ) -> None:
        super().__init__()
        cls_modules: typing.List[nn.Module] = []
        box_modules: typing.List[nn.Module] = []
        for _ in range(depth):
            cls_modules.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            cls_modules.append(nn.GroupNorm(32, in_channels))
            cls_modules.append(nn.ReLU(inplace=True))
            box_modules.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            box_modules.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            box_modules.append(nn.GroupNorm(32, in_channels))
            box_modules.append(nn.ReLU(inplace=True))

        self.cls_branch = nn.Sequential(*cls_modules)
        self.box_branch = nn.Sequential(*box_modules)
        self.cls_out = nn.Conv2d(in_channels, n_classes, 3, padding=1)
        self.box_out = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.center_out = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.apply(init_conv_std)

    def forward(self, features: typing.List[torch.Tensor]) -> NetOut:
        logits: LogitMaps = []
        boxes: BoxMaps = []
        centers: CenterMaps = []

        for feat in features:
            cls_out = self.cls_branch(feat)
            logits.append(self.cls_out(cls_out))
            centers.append(self.center_out(cls_out))
            box_out = self.box_branch(feat)
            boxes.append(self.box_out(box_out))
        return logits, centers, boxes


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

    def forward(self, features: Features) -> Features:
        return features
