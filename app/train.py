import typing as t
from torch.utils.data import DataLoader
import torch
from logging import getLogger


#  from app.models import NNModel
from tqdm import tqdm
from app.entities import Images
from app.dataset import WheatDataset, collate_fn
from app.models.detr import DETR as NNModel
from app.models.set_criterion import SetCriterion as Criterion

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(self, train_data: Images, test_data: Images,) -> None:
        num_classes = 1
        self.model = NNModel().to(device)
        self.criterion = Criterion(num_classes=num_classes).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),)
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                WheatDataset(train_data),
                shuffle=True,
                batch_size=6,
                drop_last=True,
                collate_fn=collate_fn,
            ),
            "test": DataLoader(
                WheatDataset(test_data),
                shuffle=True,
                batch_size=1,
                collate_fn=collate_fn,
            ),
        }

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            (train_loss,) = self.train_one_epoch()
            logger.info(f"{train_loss=}")

    def train_one_epoch(self) -> t.Tuple[float]:
        self.model.train()
        self.criterion.train()
        epoch_loss = 0
        count = 0
        for samples, targets in tqdm(self.data_loaders["train"]):
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return (epoch_loss / count,)
