import albumentations as A
import random
from torchvision.transforms import Normalize
from typing import *
import numpy as np
import cv2


class RandomLayout(A.DualTransform):
    def __init__(
        self,
        width: int,
        height: int,
        size_limit: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.size_limit = size_limit

    def apply(
        self, img: Any, size: Tuple[int, int], offset: Tuple[int, int], **params: dict
    ) -> Any:
        width = self.width * size[0]
        height = self.height * size[1]
        offset_x = self.width * offset[0]
        offset_y = self.height * offset[1]
        pts1 = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], 0]])
        pts2 = np.array(
            [
                [offset_x, offset_y],
                [offset_x, offset_y + height],
                [offset_x + width, offset_y],
            ]
        )
        return cv2.warpAffine(
            img,
            cv2.getAffineTransform(pts1, pts2),
            (self.width, self.height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_to_bbox(
        self,
        bbox: Any,
        size: Tuple[float, float],
        offset: Tuple[int, int],
        **params: dict
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        x1 = x1 * size[0] + offset[0]
        y1 = y1 * size[1] + offset[1]
        x2 = x2 * size[0] + offset[0]
        y2 = y2 * size[1] + offset[1]
        return x1, y1, x2, y2

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        image = params["image"]
        scale = min(
            self.height / image.shape[0], self.width / image.shape[1]
        ) * random.uniform(*self.size_limit)
        size_x = image.shape[1] * scale / self.width
        size_y = image.shape[0] * scale / self.height
        offset_x = random.uniform(0, 1.0 - size_x)
        offset_y = random.uniform(0, 1.0 - size_y)
        return {
            "size": (size_x, size_y),
            "offset": (offset_x, offset_y),
        }

    def get_params(self) -> dict:
        return {}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("width", "height", "size_limit")


normalize_mean = (0.485, 0.456, 0.406)
normalize_std = (0.229, 0.224, 0.225)

normalize = A.Normalize(mean=normalize_mean, std=normalize_std)
inv_normalize = Normalize(
    mean=[-m / s for m, s in zip(normalize_mean, normalize_std)],
    std=[1 / s for s in normalize_std],
)
