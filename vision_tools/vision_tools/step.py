import torch
from torch import nn, Tensor
from typing import Optional, Callable, Mapping, Any, TypeVar, Generic, Iterator, Tuple
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import ToDevice, Checkpoint
from .interface import MeterLike, MetricLike
from toolz import valmap


B = TypeVar("B")
T = TypeVar("T", bound=nn.Module)
CriterionFn = Callable[[T, B], Tuple[Tensor, Optional[Mapping[str, Tensor]]]]


class TrainStep(Generic[T, B]):
    def __init__(
        self,
        writer: SummaryWriter,
        criterion: CriterionFn,
        to_device: ToDevice,
        optimizer: Any,
        loader: DataLoader[B],
        meter: MeterLike,
        use_amp: bool = False,
        checkpoint: Optional[Checkpoint[T]] = None,
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.to_device = to_device
        self.use_amp = use_amp
        self.loader = loader
        self.writer = writer
        self.scaler = GradScaler()
        self.meter = meter
        self.checkpoint = checkpoint

    def __call__(self, model: T, epoch: Optional[int] = None) -> None:
        model.train()
        with autocast(enabled=self.use_amp):
            for batch in tqdm(self.loader, total=len(self.loader)):
                batch = self.to_device(**batch)
                self.optimizer.zero_grad()
                loss, other = self.criterion(model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if other is not None:
                    self.meter.update(valmap(lambda x: x.item(), other))
            for k, v in self.meter.value.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.meter.reset()

        if self.checkpoint is not None:
            self.checkpoint.save_if_needed(optimizer=self.optimizer)


EvaluateFn = Callable[[B, B], Tuple[float, Mapping[str, float]]]
InferenceFn = Callable[[T, B], B]


class EvalStep(Generic[T, B]):
    def __init__(
        self,
        writer: SummaryWriter,
        inference: InferenceFn,
        to_device: ToDevice,
        loader: DataLoader[B],
        metric: MetricLike[B],
        checkpoint: Optional[Checkpoint[T]] = None,
        metric_name: str = "eval",
    ) -> None:
        self.to_device = to_device
        self.loader = loader
        self.writer = writer
        self.metric = metric
        self.inference = inference
        self.checkpoint = checkpoint
        self.metric_name = metric_name

    @torch.no_grad()
    def __call__(self, model: T, epoch: Optional[int] = None) -> None:
        model.eval()
        for batch in tqdm(self.loader, total=len(self.loader)):
            batch = self.to_device(**batch)
            pred_batch = self.inference(model, batch)
            self.metric.update(pred_batch, batch)
        score, other = self.metric.value
        self.writer.add_scalar(f"{self.metric_name}/score", score, epoch)
        for k, v in other.items():
            self.writer.add_scalar(f"{self.metric_name}/{k}", v, epoch)
        self.metric.reset()
        if self.checkpoint is not None:
            self.checkpoint.save_if_needed(model, score)
