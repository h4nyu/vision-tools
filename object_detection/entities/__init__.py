from typing import Tuple, List, Callable
from .box import *
from .image import *
from typing_extensions import Literal


Sample = Tuple[ImageId, Image, YoloBoxes, Labels]
Batch = List[Sample]
PyramidIdx = Literal[3, 4, 5, 6, 7]

GetScore = Callable[
    [List[Tuple[ImageId, YoloBoxes, Confidences]], List[Tuple[YoloBoxes, Labels]]],
    float,
]
