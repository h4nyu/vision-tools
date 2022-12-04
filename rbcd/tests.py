import albumentations as A
import pandas as pd
import pytest
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score

from rbcd import RdcdPngDataset, SetupFold, TrainTransform, pfbeta


def test_pfbeta() -> None:
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert pfbeta(labels, predictions) == f1_score(labels, predictions)


def test_fold() -> None:
    df = pd.read_csv("/store/train.csv")
    setup_fold = SetupFold(seed=42, n_splits=5)
    folds = setup_fold(df)
    for train, test in folds:
        assert pytest.approx(
            test["cancer"].value_counts()[1] / test["cancer"].value_counts()[0], 0.1
        ) == pytest.approx(
            train["cancer"].value_counts()[1] / train["cancer"].value_counts()[0], 0.1
        )
        test_patients = set(test["patient_id"].unique())
        train_patients = set(train["patient_id"].unique())
        assert len(test_patients.intersection(train_patients)) == 0


def test_png_train_dataset() -> None:
    df = pd.read_csv("/store/train.csv")
    dataset = RdcdPngDataset(
        df.to_dict("records"),
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sample = dataset[0]
    print(sample)


def test_png_test_dataset() -> None:
    df = pd.read_csv("/store/train.csv")
    dataset = RdcdPngDataset(
        df.to_dict("records"),
        ToTensorV2(),
        image_dir="/store/rsna-breast-cancer-256-pngs",
    )
    sample = dataset[0]
    print(sample)


def test_eda() -> None:
    df = pd.read_csv("/store/train.csv")
    print(df["density"])
