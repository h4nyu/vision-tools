from typing import NewType
from torch import Tensor, ByteTensor, FloatTensor

Image = NewType("Image", Tensor)  # [C, H, W] dtype
ImageBatch = NewType("ImageBatch", Tensor)  # [B, C, H, W]
ImageSize = tuple[int, int]  # W, H
RGB = tuple[int, int, int]
