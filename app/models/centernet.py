import typing as t
from app import config
from torch import nn, Tensor
from .modules import ConvBR2d, SENextBottleneck2d
from .matcher import Outputs


class BoxRegression(nn.Module):
    def __init__(self, in_channels: int, num_queries: int) -> None:
        super().__init__()
        self.conv0 = ConvBR2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_queries * 4,
                kernel_size=1,
                padding=0,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        self.num_queries = num_queries

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.conv0(x)
        x = self.act(x)
        x = self.out(x)
        x = x.view(b, self.num_queries, 4)
        return x


class LabelClassification(nn.Module):
    def __init__(self, in_channels: int, num_queries: int, num_classes: int) -> None:
        super().__init__()
        self.conv0 = ConvBR2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_queries * num_classes,
                kernel_size=1,
                padding=0,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.conv0(x)
        x = self.act(x)
        x = self.out(x)
        x = x.view(b, self.num_queries, self.num_classes).sigmoid()
        return x


class Down2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels: int, out_channels: int, pool: t.Literal["max", "avg"] = "max"
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            SENextBottleneck2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                is_shortcut=True,
                pool=pool,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CenterNet(nn.Module):
    def __init__(self, num_queries: int=config.num_queries, num_classes: int=config.num_classes) -> None:
        super().__init__()
        base_channels = 16
        self.box_reg = BoxRegression(
            in_channels=base_channels * 2, num_queries=num_queries,
        )
        self.label_clf = LabelClassification(
            in_channels=base_channels * 2,
            num_queries=num_queries,
            num_classes=num_classes + 1,
        )
        self.inc = nn.Conv2d(
            in_channels=3,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.down0 = Down2d(in_channels=base_channels, out_channels=base_channels * 2)
        self.down1 = Down2d(
            in_channels=base_channels * 2, out_channels=base_channels * 2
        )
        self.down2 = Down2d(
            in_channels=base_channels * 2, out_channels=base_channels * 2
        )
        self.down3 = Down2d(
            in_channels=base_channels * 2, out_channels=base_channels * 2
        )

    def forward(self, x: Tensor) -> Outputs:
        x = self.inc(x)
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        reg = self.box_reg(x)
        clf = self.label_clf(x)
        out: Outputs = {
            "pred_logits": clf,  # last layer output
            "pred_boxes": reg,  # last layer output
        }
        return out
