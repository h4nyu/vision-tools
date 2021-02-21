from typing import Union
from torch import Tensor

FP = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]  # p3, p4, p5, p6, p7
Number = Union[float, int]

from .box import *
from .image import *
from .point import *
