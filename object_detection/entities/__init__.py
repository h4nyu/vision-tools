from typing import Tuple, List
from .box import *
from .image import *
from typing_extensions import Literal


Sample = Tuple[ImageId, Image, YoloBoxes, Labels]
Batch = List[Sample]
PyramidIdx = Literal[3, 4, 5, 6, 7]
