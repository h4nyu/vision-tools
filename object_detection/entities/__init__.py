from typing import Tuple, List
from .box import CoCoBoxes, Labels, PredBoxes, LabelBoxes, Confidences, YoloBoxes
from .image import Image, ImageBatch, ImageId, ImageSize
from typing_extensions import Literal


Sample = Tuple[ImageId, Image, YoloBoxes]
Batch = List[Sample]
PyramidIdx = Literal[3, 4, 5, 6, 7]
