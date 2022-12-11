from collections import namedtuple
from types import SimpleNamespace

import albumentations as A
import pandas as pd
import pytest
import torch
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from rbcd import (
    Model,
    OverBatchSampler,
    OverSampler,
    RdcdPngDataset,
    SetupFolds,
    TrainTransform,
    UnderBatchSampler,
    pfbeta,
)


def test_pfbeta() -> None:
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert pfbeta(labels, predictions) == f1_score(labels, predictions)


def test_fold() -> None:
    df = pd.read_csv("/store/train.csv")
    setup_fold = SetupFolds(seed=42, n_splits=5)
    folds = setup_fold(df)
    for train, test in folds:
        assert pytest.approx(
            test["cancer"].value_counts()[1] / test["cancer"].value_counts()[0], 0.1
        ) == pytest.approx(
            train["cancer"].value_counts()[1] / train["cancer"].value_counts()[0], 0.1
        )
        print(
            test["cancer"].value_counts()[1] / test["cancer"].value_counts()[0],
            train["cancer"].value_counts()[1] / train["cancer"].value_counts()[0],
        )
        test_patients = set(test["patient_id"].unique())
        train_patients = set(train["patient_id"].unique())
        assert len(test_patients.intersection(train_patients)) == 0


def test_png_train_dataset() -> None:
    df = pd.read_csv("/store/train.csv")
    cfg = SimpleNamespace(
        hflip=0.5,
        vflip=0.0,
        scale_limit=0.0,
        rotate_limit=0,
        border_mode=0,
    )
    image_size = 512
    dataset = RdcdPngDataset(
        df,
        TrainTransform(cfg),
        image_dir=f"/store/rsna-breast-cancer-{image_size}-pngs",
    )
    batch_size = 16
    batch_sampler = UnderBatchSampler(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    batch = next(iter(dataloader))
    assert batch["target"].sum() == batch_size // 2
    assert batch["image"].shape == (batch_size, 1, image_size, image_size)
    assert batch["target"].shape == (batch_size, 1)
    grid = make_grid(batch["image"], nrow=batch_size // 2)
    print(batch["image_id"])
    save_image(grid, "/test_output/grid.png")
    # save_image(sample["image"], "/test_output/sample.png")


def test_balanced_batch_sampler() -> None:
    df = pd.read_csv("/store/train.csv").loc[:100]
    non_cancer = df[df["cancer"] == 0]
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sampler = UnderBatchSampler(dataset, batch_size=8, shuffle=False)
    assert len(sampler) == len(non_cancer) // (8 // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    for batch in dataloader:
        assert batch["target"].sum() == 4
        assert batch["target"].shape == (8, 1)


def test_balanced_batch_down_sampler() -> None:
    df = pd.read_csv("/store/train.csv")[:1000]
    cancer = df[df["cancer"] == 1]
    batch_size = 8
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sampler = UnderBatchSampler(dataset, batch_size=batch_size, shuffle=False)
    assert len(sampler) == len(cancer) // (8 // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    for batch in dataloader:
        assert batch["target"].sum() == 4
        assert batch["target"].shape == (8, 1)


def test_over_sampler() -> None:
    df = pd.read_csv("/store/train.csv")[:100]
    cancer = df[df["cancer"] == 0]
    batch_size = 8
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sampler = OverSampler(dataset, shuffle=True, ratio=1 / 1)
    assert len(sampler) == len(cancer) * 2
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    for batch in dataloader:
        assert batch["target"].sum() > 0
        assert batch["target"].shape == (8, 1)


def test_png_test_dataset() -> None:
    df = pd.read_csv("/store/train.csv")
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 256, 256)


def test_eda() -> None:
    df = pd.read_csv("/store/train.csv")
    print(df["density"])


def test_model() -> None:
    model = Model(
        name="tf_efficientnet_b3_ns",
        in_channels=1,
    )
    image = torch.randn(1, 1, 256, 256)
    output = model(image)
    assert output.shape == (1,)
