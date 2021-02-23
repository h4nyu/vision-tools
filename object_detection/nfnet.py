import torch
from torch import nn, Tensor
from torch.functional import F


class WSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride:int=1,
        padding:int=0,
        dilation:int=1,
        groups:int=1,
        bias:bool=True,
        padding_mode:str="zeros",
        eps:float=1e-4,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        nn.init.kaiming_normal_(self.weight)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weight(self) -> Tensor:
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[0:]))
        scale = torch.rsqrt(
            torch.max(var * fan_in, torch.tensor(self.eps).to(var.device))
        ) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        print(shift)
        return self.weight * scale - shift

    def forward(self, input:Tensor) -> Tensor:
        weight = self.standardize_weight()
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
