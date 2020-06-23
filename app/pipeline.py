import torch
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, ConcatDataset
from app.models.centernet import collate_fn, CenterNet, Visualize, Trainer
from app.dataset.wheat import WheatDataset
from app.model_loader import ModelLoader
from app import config
from app.preprocess import kfold


def train(fold_idx: int) -> None:
    dataset = WheatDataset(annot_file=config.annot_file, max_size=config.max_size,)
    fold_keys = [x[2].shape[0] // 20 for x in dataset.rows]
    train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[
        fold_idx
    ]
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    out_dir = f"/kaggle/input/models/{fold_idx}"
    model = CenterNet()
    model_loader = ModelLoader(out_dir=out_dir, model=model)
    visualize = Visualize("./", "centernet", limit=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    trainer = Trainer(
        train_loader, test_loader, model_loader, optimizer, visualize, config.device
    )
    trainer.train(100)


def submit() -> None:
    ...
