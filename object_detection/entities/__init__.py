from typing import Tuple, List, Callable
from .box import *
from .image import *
from typing_extensions import Literal


TrainSample = Tuple[ImageId, Image, YoloBoxes, Labels]
PyramidIdx = Literal[3, 4, 5, 6, 7]
FP = Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor
]  # p3, p4, p5, p6, p7

SideChannels = Tuple[int, int, int, int, int]  # p3, p4, p5, p6, p7

PredictionSample = Tuple[ImageId, Image]
