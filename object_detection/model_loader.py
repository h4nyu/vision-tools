import torch
from logging import getLogger
from torch import nn
from pathlib import Path
from typing import Dict, Tuple
import json

logger = getLogger(__name__)


class ModelLoader:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)
        self.checkpoint_file = self.out_dir / "checkpoint.json"
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def check_point_exists(self) -> bool:
        return self.checkpoint_file.exists()

    def load(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        logger.info(f"load model from {self.out_dir}")
        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)
        model.load_state_dict(
            torch.load(self.out_dir / f"model.pth")  # type: ignore
        )
        return model, data

    def save(self, model: nn.Module, metrics: Dict[str, float] = {}) -> None:
        with open(self.checkpoint_file, "w") as f:
            json.dump(metrics, f)
        logger.info(f"save model to {self.out_dir}")
        torch.save(model.state_dict(), self.out_dir / f"model.pth")  # type: ignore
