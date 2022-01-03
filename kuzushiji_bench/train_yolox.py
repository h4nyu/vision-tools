import os
import torch_optimizer as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, Inference
from kuzushiji_bench.yolox import get_model, get_criterion
from vision_tools.meter import MeanReduceDict
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch
from kuzushiji_bench.data import (
    KuzushijiDataset,
    TrainTransform,
    Transform,
    read_train_rows,
    collate_fn,
    kfold,
)
from kuzushiji_bench.metric import Metric
from tqdm import tqdm


def main() -> None:
    seed_everything()
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    writer = SummaryWriter(os.path.join("runs", cfg.name))
    checkpoint = Checkpoint[YOLOX](
        root_path=os.path.join(cfg.root_dir, cfg.name),
        default_score=0.0,
    )
    model = get_model(cfg)
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    criterion = get_criterion(cfg)
    optimizer = optim.AdaBound(model.parameters(), **cfg.optimizer)
    annotations = read_train_rows(cfg.root_dir)
    train_rows, validation_rows = kfold(annotations, **cfg.fold)
    train_dataset = KuzushijiDataset(
        train_rows[:200],
        transform=TrainTransform(cfg.image_size),
    )
    val_dataset = KuzushijiDataset(
        validation_rows[:100],
        transform=Transform(cfg.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg.train_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg.val_loader,
    )
    to_device = ToDevice(cfg.device)

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=to_device,
        criterion=criterion,
        optimizer=optimizer,
        loader=train_loader,
        meter=MeanReduceDict(),
        writer=writer,
    )
    metric = Metric()

    eval_step = EvalStep[YOLOX, TrainBatch](
        to_device=to_device,
        loader=val_loader,
        metric=metric,
        writer=writer,
        inference=Inference(),
        checkpoint=checkpoint,
    )

    for epoch in range(cfg.num_epochs):
        train_step(model, epoch)
        eval_step(model, epoch)


if __name__ == "__main__":
    main()
