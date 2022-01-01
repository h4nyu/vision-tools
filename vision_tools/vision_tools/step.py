from torch import nn, Tensor
from typing import Optional, Callable, Mapping, Any, TypeVar, Generic, Iterator
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import ToDevice
from .interface import MeterLike
from toolz import valmap


B = TypeVar("B")
T = TypeVar("T", bound=nn.Module)
CriterionFn = Callable[[T, B], tuple[Tensor, Optional[Mapping[str, Tensor]]]]


class TrainStep(Generic[T, B]):
    def __init__(
        self,
        writer: SummaryWriter,
        criterion: CriterionFn,
        to_device: ToDevice,
        optimizer: Any,
        loader: DataLoader[B],
        meter: MeterLike,
        use_amp: bool = True,
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.to_device = to_device
        self.use_amp = use_amp
        self.loader = loader
        self.writer = writer
        self.scaler = GradScaler()
        self.meter = meter

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
                    self.meter.accumulate(valmap(lambda x: x.item(), other))
            self.writer.add_scalars("train", self.meter.value, epoch)
            self.meter.reset()

