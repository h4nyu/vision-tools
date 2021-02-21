from torch import nn, Tensor
from object_detection.bifpn import BiFPN
from object_detection.centernet import Head
from object_detection.image import ImageBatch


class Net(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        backbone: nn.Module,
        cls_depth: int = 1,
        fpn_depth: int = 1,
        out_idx: int = 4,
    ) -> None:
        super().__init__()
        self.out_idx = out_idx - 3
        self.channels = channels
        self.backbone = backbone
        self.fpn = nn.Sequential(*[BiFPN(channels=channels) for _ in range(fpn_depth)])
        self.hm_reg = nn.Sequential(
            Head(
                in_channels=channels,
                out_channels=num_classes,
                depth=cls_depth,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: ImageBatch) -> Tensor:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmaps = self.hm_reg(fp[self.out_idx])
        return heatmaps
