import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from app.train import Trainer, Preditor
from app.models.centernet import Evaluate
from app.preprocess import load_lables, KFold
from app import config
from app.utils import DetectionPlot
from app.models.utils import box_xyxy_to_cxcywh
from albumentations.pytorch.transforms import ToTensorV2


def eda_bboxes() -> None:
    images = load_lables()
    box_counts = pipe(images.values(), map(lambda x: len(x.boxes)), list)
    max_counts = max(box_counts)
    print(f"{max_counts=}")
    min_counts = min(box_counts)
    print(f"{min_counts=}")
    mean_counts = np.mean(box_counts)
    print(f"{mean_counts=}")
    ws = pipe(images.values(), map(lambda x: x.width), list)
    max_width = max(ws)
    print(f"{max_width=}")
    min_width = min(ws)
    print(f"{min_width=}")


def train(fold_idx: int) -> None:
    images = load_lables()
    kf = KFold()
    train_data, test_data = list(kf(images))[fold_idx]
    trainer = Trainer(
        train_data, test_data, Path(config.root_dir).joinpath(str(fold_idx))
    )
    trainer.train(1000)


def pre_submit(fold_idx: int) -> None:
    images = load_lables()
    evaluate = Evaluate()
    p = Preditor(Path(config.root_dir).joinpath(str(fold_idx)), images,)
    preds = p()

    score = evaluate(preds, images)
    print(f"{score=}")

    plot = DetectionPlot()
    to_tensor = ToTensorV2()
    sample_key = next(iter(preds.keys()))
    sample = preds[sample_key]
    gt = images[sample_key]

    plot.with_image(to_tensor(image=sample.get_img())['image'])
    plot.with_boxes(sample.boxes, sample.confidences, color="red")
    plot.with_boxes(box_xyxy_to_cxcywh(gt.boxes), color="blue")
    plot.save('/store/plot/sample.png')
