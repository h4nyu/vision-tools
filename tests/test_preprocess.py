import typing as t
from app.preprocess import load_lables, KFold


def test_load_lables() -> None:
    images = load_lables()
    first_sample = next(iter(images))
    assert len(first_sample.boxes) == 47


def test_kfold() -> None:
    images = load_lables()
    kf = KFold()
    fold_train, fold_valid = next(kf(images))
    assert len(fold_train) + len(fold_valid) == len(images)
