import torch
from typing import Union, Optional, Callable, Any
from torch import nn, optim, Tensor
from torch.optim.optimizer import Optimizer, _params_t

from collections.abc import Iterable


def unitwise_norm(x: Tensor) -> Tensor:
    if x.ndim <= 1:
        dim: Union[int, list[int]] = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError(f"Wrong input dimensions: {x.ndim}")

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


class SGD_AGC(Optimizer):
    def __init__(
        self,
        named_params: _params_t,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        clipping: float = None,
        eps: float = 1e-3,
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            # Extra defaults
            clipping=clipping,
            eps=eps,
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Put params in list so each one gets its own group
        params = []
        for name, param in named_params:
            params.append({"params": param, "name": name})

        super(SGD_AGC, self).__init__(params, defaults)

    def __setstate__(self, state: Any) -> None:
        super(SGD_AGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            # Extra values for clipping
            clipping = group["clipping"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                # =========================
                # Gradient clipping
                if clipping is not None:
                    param_norm = torch.maximum(
                        unitwise_norm(p), torch.tensor(eps).to(p.device)
                    )
                    grad_norm = unitwise_norm(d_p)
                    max_norm = param_norm * group["clipping"]

                    trigger_mask = grad_norm > max_norm
                    clipped_grad = p.grad * (
                        max_norm
                        / torch.maximum(grad_norm, torch.tensor(1e-6).to(p.device))
                    )
                    d_p = torch.where(trigger_mask, clipped_grad, d_p)
                # =========================

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
