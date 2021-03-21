from typing import *
from torch import Tensor, ByteTensor, FloatTensor

Image = NewType("Image", Tensor)  # [C, H, W] dtype
ImageBatch = NewType("ImageBatch", Tensor)  # [B, C, H, W]
ImageSize = Tuple[int, int]  # W, H
RGB = Tuple[int, int, int]


def inv_scale_and_pad(
    original: Tuple[int, int], padded: Tuple[int, int]
) -> Tuple[float, Tuple[float, float]]:
    original_w, original_h = original
    padded_w, padded_h = padded
    original_longest = max(original)
    if original_longest == original_w:
        scale = original_longest / padded_w
        pad = (padded_h - original_h / scale) / 2
        return scale, (0, pad)
    else:
        scale = original_longest / padded_h
        pad = (padded_w - original_w / scale) / 2
        return scale, (pad, 0)
