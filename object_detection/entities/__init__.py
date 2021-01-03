from typing import Tuple, List, Callable
from .box import *
from .image import *
from typing_extensions import Literal


TrainSample = Tuple[ImageId, Image, PascalBoxes, Labels]
FP = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]  # p3, p4, p5, p6, p7

SideChannels = Tuple[int, int, int, int, int]  # p3, p4, p5, p6, p7

PredictionSample = Tuple[ImageId, Image]
