import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor, nn, optim
from torch.nn import functional as F
from pytorch_metric_learning.losses import ArcFaceLoss
import timm
from typing import Any


class ConvNeXt(nn.Module):
    def __init__(self, name: str, embedding_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained, num_classes=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        out = F.normalize(out, p=2, dim=1)
        return out


class Lite(LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = ConvNeXt(
            name=cfg["model_name"],
            embedding_size=cfg["embedding_size"],
            pretrained=cfg["pretrained"],
        )
        self.loss_fn = ArcFaceLoss(
            num_classes=cfg["num_classes"],
            embedding_size=cfg["embedding_size"],
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"])

    def forward(self, batch: Any) -> Tensor: # type: ignore
        x, y = batch
        embeddings = self.model(x)
        loss = self.loss_fn(embeddings, y)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor: # type: ignore
        loss = self(batch)
        self.log("train_loss", loss)
        return loss
