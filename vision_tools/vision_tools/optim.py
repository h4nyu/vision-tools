from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor
from torch.optim import Optimizer


# WIP save and load
class Lookahead(Optimizer):
    r"""Implements Lookahead optimizer.

    It's been proposed in paper: Lookahead Optimizer: k steps forward, 1 step back
    (https://arxiv.org/pdf/1907.08610.pdf)

    Args:
        optimizer: The optimizer object used in inner loop for fast weight updates.
        alpha:     The learning rate for slow weight update.
                   Default: 0.5
        k:         Number of iterations of fast weights updates before updating slow
                   weights.
                   Default: 5

    Example:
        > optim = Lookahead(optimizer)
        > optim = Lookahead(optimizer, alpha=0.6, k=10)
    """

    def __init__(self, optimizer: Optimizer, alpha: float = 0.5, k: int = 5) -> None:
        assert 0.0 <= alpha <= 1.0
        assert k >= 1
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.k_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.slow_weights = [
            [param.clone().detach() for param in group["params"]]
            for group in self.param_groups
        ]

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = self.optimizer.step(closure)
        self.k_counter += 1
        if self.k_counter >= self.k:
            for group, slow_weight in zip(self.param_groups, self.slow_weights):
                for param, weight in zip(group["params"], slow_weight):
                    weight.data.add_((param.data - weight.data), alpha=self.alpha)
                    param.data.copy_(weight.data)
            self.k_counter = 0
        return loss

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "slow_weights": self.slow_weights,
            "alpha": self.alpha,
            "k": self.k,
            "k_counter": self.k_counter,
        }

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
