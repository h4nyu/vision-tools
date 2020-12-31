import typing as t
from typing import NewType, Tuple
from torch import Tensor, ByteTensor, FloatTensor

ImageId = NewType("ImageId", str)
Image = NewType("Image", Tensor)  # [C, H, W] dtype

ImageBatch = NewType("ImageBatch", Tensor)  # [B, C, H, W]
ImageSize = Tuple[int, int]  # H, W
RGB = Tuple[int, int, int]
