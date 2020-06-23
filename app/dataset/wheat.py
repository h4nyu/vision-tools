import typing as t
import numpy as np
import pandas as pd
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
import albumentations as albm
from glob import glob
from app.entities.box import CoCoBoxes, coco_to_yolo
from app.entities.image import ImageSize, ImageId, Image
from app.entities import Sample
from albumentations.pytorch.transforms import ToTensorV2
from skimage.io import imread
from cytoolz.curried import pipe, groupby, valmap
from typing_extensions import Literal


from pathlib import Path
from app import config
from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

Row = t.Tuple[Tensor, Tensor, str]
Batch = t.Sequence[Row]


def parse_boxes(strs: t.List[str]) -> CoCoBoxes:
    coco = CoCoBoxes(
        torch.tensor(np.stack([np.fromstring(s[1:-1], sep=",") for s in strs]))
    )
    return coco


def load_lables(
    annot_file: str, limit: t.Optional[int] = None
) -> t.Sequence[t.Tuple[ImageId, ImageSize, CoCoBoxes]]:
    df = pd.read_csv(annot_file, nrows=limit)
    df["bbox"] = df["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
    return df.groupby("image_id").apply(
        lambda x: (
            ImageId(x["image_id"].iloc[0]),
            (x["width"].iloc[0], x["height"].iloc[0]),
            (torch.tensor(np.stack(x["bbox"]))),
        )
    )


def get_img(image_id: ImageId) -> t.Any:
    image_path = f"{config.image_dir}/{image_id}.jpg"
    return (imread(image_path) / 255).astype(np.float32)


class WheatDataset(Dataset):
    def __init__(
        self,
        annot_file: str,
        max_size: int = 512,
        mode: Literal["train", "test"] = "train",
    ) -> None:
        super().__init__()
        self.annot_file = Path(annot_file)
        self.rows = load_lables(annot_file)
        self.mode = mode
        self.cache: t.Dict[str, t.Any] = dict()
        self.image_dir = Path(config.image_dir)

        bbox_params = {"format": "coco", "label_fields": ["labels"]}
        self.pre_transforms = albm.Compose(
            [albm.LongestMaxSize(max_size=max_size),], bbox_params=bbox_params,
        )
        self.train_transforms = albm.Compose(
            [
                albm.VerticalFlip(),
                albm.RandomRotate90(),
                albm.HorizontalFlip(),
                albm.RandomBrightness(limit=0.1),
            ],
            bbox_params=bbox_params,
        )

        self.post_transforms = albm.Compose([ToTensorV2(),])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        image_id, _, boxes = self.rows[index]
        image = get_img(image_id)
        labels = np.zeros(boxes.shape[0])
        res = self.pre_transforms(image=image, bboxes=boxes, labels=labels)
        if self.mode == "train":
            res = self.train_transforms(**res)
        res = self.post_transforms(**res)

        image = Image(res["image"].float())
        _, h, w = image.shape
        boxes = CoCoBoxes(torch.tensor(res["bboxes"]).float())
        yolo_boxes = coco_to_yolo(boxes, (w, h))
        return image_id, image, yolo_boxes


#
#  class PreditionDataset(Dataset):
#      def __init__(self, image_dir: str = config.test_image_dir,) -> None:
#          print(f"{config.test_image_dir}/*.jpg")
#          rows: t.List[t.Tuple[str, Path]] = []
#          for p in glob(f"{config.test_image_dir}/*.jpg"):
#              path = Path(p)
#              rows.append((path.stem, path))
#          self.rows = rows
#
#      def __len__(self) -> int:
#          return len(self.rows)
#
#      def __getitem__(self, index: int) -> t.Tuple[Tensor, str]:
#          id, path = self.rows[index]
#          image = (imread(path) / 255).astype(np.float32)
#          image = transforms(image=image)["image"].float()
#          return image, id
