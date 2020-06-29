import torch
import numpy as np
import typing as t
from torch import nn, Tensor
from object_detection.entities.box import YoloBoxes, PascalBoxes, pascal_to_yolo
from object_detection.entities import PyramidIdx


class Anchors:
    def __init__(
        self,
        size: int=4,
        stride: int = 1,
        ratios: t.List[float] = [0.5, 1, 2],
        scales: t.List[float] = [1.0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    ) -> None:
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)

    def __call__(self, image: Tensor) -> YoloBoxes:
        """
        image: [B, C, H, W]
        return: [B, num_anchors, 4]
        """
        _, _, h, w = image.shape
        image_shape = np.array((h, w))
        anchors = generate_anchors(
            base_size=self.size, ratios=self.ratios, scales=self.scales
        )
        all_anchors = shift(image_shape, self.stride, anchors).astype(np.float32)
        return pascal_to_yolo(
            PascalBoxes(torch.from_numpy(all_anchors).to(image.device)), (w, h)
        )


def shift(shape: t.Any, stride: int, anchors: t.Any) -> t.Any:
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2)
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(
    base_size: int, ratios: t.List[float], scales: t.List[float],
) -> t.Any:
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors
