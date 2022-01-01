from torch import nn, Tensor
from typing import Optional, Callable, Mapping, Any, TypeVar, Generic, Iterator
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from .utils import ToDevice
from toolz import valmap


B = TypeVar("B")
T = TypeVar("T", bound=nn.Module)
CriterionFn = Callable[[T, B], tuple[Tensor, Optional[Mapping[str, Tensor]]]]


class TrainStep(Generic[T, B]):
    def __init__(
        self,
        criterion: CriterionFn,
        to_device: ToDevice,
        optimizer: Any,
        loader: Iterator[B],
        use_amp: bool = True,
        batch_interval: int = 100,
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.to_device = to_device
        self.use_amp = use_amp
        self.loader = loader
        self.scaler = GradScaler()
        self.batch_interval = batch_interval

    def __call__(self, model: T) -> None:
        model.train()
        with autocast(enabled=self.use_amp):
            for batch in tqdm(self.loader, total=self.batch_interval):
                batch = self.to_device(**batch)
                self.optimizer.zero_grad()
                loss, other = self.criterion(model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if other is not None:
                    valmap(lambda x: x.item(), other)
