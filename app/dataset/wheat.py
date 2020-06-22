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
) -> t.List[t.Tuple[ImageId, ImageSize, CoCoBoxes]]:
    df = pd.read_csv(annot_file, nrows=limit)
    rows = pipe(
        df.to_dict("records"),
        groupby(lambda x: x["image_id"]),
        valmap(
            lambda x: (
                ImageId(x[0]["image_id"]),
                ((x[0]["width"], x[0]["height"])),
                parse_boxes([b["bbox"] for b in x]),
            )
        ),
        lambda x: x.values(),
        list,
    )
    return rows


train_transforms = albm.Compose(
    [
        albm.VerticalFlip(),
        albm.RandomRotate90(),
        albm.HorizontalFlip(),
        albm.RandomBrightness(limit=0.1),
    ],
    bbox_params=dict(format="albumentations", label_fields=["labels"]),
)


transforms = albm.Compose(
    [
        #  albm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def get_img(image_id: ImageId) -> t.Any:
    image_path = f"{config.image_dir}/{image_id}.jpg"
    return (imread(image_path) / 255).astype(np.float32)


class WheatDataset(Dataset):
    def __init__(
        self,
        annot_file: str,
        mode: t.Literal["train", "test"] = "train",
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        self.annot_file = Path(annot_file)
        self.rows = load_lables(annot_file)
        self.mode = mode
        self.cache: t.Dict[str, t.Any] = dict()
        self.use_cache = use_cache
        self.image_dir = Path(config.image_dir)
        self.post_transforms = albm.Compose([ToTensorV2(),])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        image_id, image_size, coco = self.rows[index]
        image = get_img(image_id)
        labels = torch.zeros(len(coco)).long()
        if self.mode == "train":
            res = train_transforms(image=image, bboxes=coco, labels=labels)
            boxes = torch.tensor(res["bboxes"], dtype=torch.float32)

        image = res["image"]
        image = self.post_transforms(image=image)["image"]
        boxes = CoCoBoxes(torch.tensor(res["bboxes"]))
        image = transforms(image=image)["image"].float()
        yolo_boxes = coco_to_yolo(boxes, image_size)
        return image_id, Image(image), yolo_boxes


class PreditionDataset(Dataset):
    def __init__(self, image_dir: str = config.test_image_dir,) -> None:
        print(f"{config.test_image_dir}/*.jpg")
        rows: t.List[t.Tuple[str, Path]] = []
        for p in glob(f"{config.test_image_dir}/*.jpg"):
            path = Path(p)
            rows.append((path.stem, path))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, str]:
        id, path = self.rows[index]
        image = (imread(path) / 255).astype(np.float32)
        image = transforms(image=image)["image"].float()
        return image, id
