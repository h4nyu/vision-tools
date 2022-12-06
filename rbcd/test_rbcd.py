import albumentations as A
import pandas as pd
import pytest
import torch
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from rbcd import (
    BalancedBatchSampler,
    Model,
    RdcdPngDataset,
    SetupFolds,
    TrainTransform,
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
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 256, 256)


def test_balanced_batch_sampler() -> None:
    df = pd.read_csv("/store/train.csv").loc[:100]
    non_cancer = df[df["cancer"] == 0]
    dataset = RdcdPngDataset(
        df,
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sampler = BalancedBatchSampler(dataset, batch_size=8, shuffle=False)
    assert len(sampler) == len(non_cancer) // (8 // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    for batch in dataloader:
        assert batch["target"].sum() == 4


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
        num_classes=1,
        in_channels=1,
    )
    image = torch.randn(1, 1, 256, 256)
    output = model(image)
    assert output.shape == (1, 1)
