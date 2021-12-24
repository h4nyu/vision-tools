import torch
from typing import *
from torch import Tensor
from PIL import ImageDraw
import json
import random
import numpy as np
import torch.nn.functional as F
from typing import Optional
from torch import nn
from pathlib import Path
from torch import Tensor
from logging import getLogger
from vision_tools import (
    Number,
    resize_points,
)
from torchvision.utils import save_image
from torch.nn.functional import interpolate

logger = getLogger(__name__)


def init_seed(seed: int = 777) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


colors = [
    "green",
    "red",
    "blue",
]
