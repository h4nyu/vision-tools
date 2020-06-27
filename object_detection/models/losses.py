import typing as t
import numpy as np
import torch.nn.functional as F
from typing_extensions import Literal
from torch import Tensor
import torch
import torch.nn as nn

from typing import Dict

Reduction = Literal["none", "mean", "sum"]
