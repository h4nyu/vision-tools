from __future__ import annotations

import timm
from pytorch_metric_learning.losses import ArcFaceLoss
from toolz.curried import map, pipe
from torch import Tensor, nn
from torch.nn import functional as F

from hwad_bench.data import Annotation, HwadCropedDataset
from vision_tools.utils import Checkpoint, seed_everything


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


def train(
    dataset_cfg: dict,
    model_cfg: dict,
    fold: int,
    annotations: list[Annotation],
    image_dir: str,
) -> None:
    seed_everything()
    loss_fn = ArcFaceLoss(
        num_classes=dataset_cfg["num_classes"],
        embedding_size=model_cfg["embedding_size"],
    )
    model = get_model(model_cfg)
    checkpoint = get_checkpoint(model_cfg)
    saved_state = checkpoint.load("best")
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
    dataset = HwadCropedDataset(
        rows=annotations,
        image_dir=image_dir,
    )
