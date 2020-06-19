import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from app.train import Trainer, Preditor
from app.eval import Evaluate
from app.preprocess import load_lables, KFold
from app import config
from app.utils import DetectionPlot
from app.models.utils import box_xyxy_to_cxcywh
from app.submit import save_csv
from albumentations.pytorch.transforms import ToTensorV2


def eda_bboxes() -> None:
    images = load_lables()
    box_counts = pipe(images, map(lambda x: len(x.boxes)), list)
    max_counts = max(box_counts)
    print(f"{max_counts=}")
    min_counts = min(box_counts)
    print(f"{min_counts=}")
    mean_counts = np.mean(box_counts)
    print(f"{mean_counts=}")
    ws = pipe(images, map(lambda x: x.w), list)
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


def submit(fold_idx: int) -> None:
    p = Preditor(Path(config.root_dir).joinpath(str(fold_idx)))
    images, preds = p()
    for i, (img, boxes) in enumerate(zip(images, preds)):
        plot = DetectionPlot(figsize=(6, 6))
        plot.with_image(img.detach().cpu())
        plot.with_boxes(boxes.boxes, boxes.confidences, color="red")
        plot.save(f"/kaggle/output/pred-{i}.png")
    save_csv(preds, "/kaggle/output/submission.csv")
