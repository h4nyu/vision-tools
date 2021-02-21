from torch import Tensor
from typing import NewType

Points = NewType("Points", Tensor)  # [B, 2] [x, y]
