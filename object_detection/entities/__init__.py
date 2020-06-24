from typing import Tuple, List
from .box import CoCoBoxes, Labels, PredBoxes, LabelBoxes, Confidences, YoloBoxes
from .image import Image, ImageBatch, ImageId, ImageSize

Sample = Tuple[ImageId, Image, YoloBoxes]
Batch = List[Sample]
