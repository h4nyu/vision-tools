import typing as t
import torchvision
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from .utils import NestedTensor
from .transformer import Transformer
from .matcher import Outputs
from .position_embedding import PositionEmbeddingSine


class Joiner(nn.Module):
    def __init__(self, model0: nn.Module, model1: nn.Module) -> None:
        super().__init__()
        self.model0 = model0
        self.model1 = model1

    def forward(
        self, tensor_list: NestedTensor
    ) -> t.Tuple[t.List[NestedTensor], t.List[Tensor]]:
        xs = self.model0(tensor_list)
        out: t.List[NestedTensor] = []
        pos: t.List[Tensor] = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.model1(x).to(x.tensors.dtype))

        return out, pos


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


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h_dims, h_dims + [output_dim])
        )
        self.act = nn.ReLU(inplace=True)
        self._init_bias()

    @torch.no_grad()
    def _init_bias(self) -> None:
        """
        to centerize box positions
        """
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight,)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        backbone = Backbone("resnet18")
        self.backbone = Joiner(backbone, PositionEmbeddingSine(hidden_dim // 2))

        self.transformer = Transformer(d_model=hidden_dim,)
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, samples: NestedTensor) -> Outputs:
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs, _ = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out: Outputs = {
            "pred_logits": outputs_class[-1],  # last layer output
            "pred_boxes": outputs_coord[-1],  # last layer output
        }
        return out
