import typing as t
import torchvision
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from .utils import NestedTensor


BackBoneOutputs = t.Dict[str, NestedTensor]


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ) -> None:
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> BackBoneOutputs:
        xs = self.body(tensor_list.tensors)
        out: BackBoneOutputs = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(
        self, name: str, train_backbone: bool = True, return_interm_layers: bool = True,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, False], pretrained=True,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

    def forward(self, tensor_list: NestedTensor) -> BackBoneOutputs:
        xs = self.body(tensor_list.tensors)
        out: BackBoneOutputs = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class DETR(nn.Module):
    def __init__(self, num_classes: int = 2, num_queries: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = Backbone("resnet34")
        #  self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples: NestedTensor) -> None:
        features, pos = self.backbone(samples)

        ...
