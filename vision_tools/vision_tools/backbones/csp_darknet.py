from torch import nn
from typing import Callable
from ..block import DefaultActivation, DWConv, ConvBnAct


class CSPDarknet(nn.Module):
    def __init__(
        self, depthwise: bool = False, act: Callable = DefaultActivation
    ) -> None:
        super().__init__()
        Conv = DWConv if depthwise else ConvBnAct
