import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..bus import TokenBus
from .base import Block, register


@register("tcn")
class TCNBlock(Block):
    """Two causal dilated convolutions (Bai et al. 2018). Native format CHF = [B, d, L].

    Adapter in:  zero padded tokens (conv has no mask), transpose [B, L, d] -> [B, d, L], left-pad for causality.
    Adapter out: transpose back to [B, L, d]. Position stays implicit in the kernel.
    """

    native_format = "CHF"

    def __init__(self, d: int, dropout: float = 0.1, kernel_size: int = 3, dilation: int = 1, **kwargs):
        super().__init__(d, dropout)
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(d, d, kernel_size, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(d, d, kernel_size, dilation=dilation))
        self.act = nn.ReLU()
        self.inner_drop = nn.Dropout(dropout)

    def mix(self, bus: TokenBus) -> torch.Tensor:
        h = bus.zero_padding().tokens.transpose(1, 2)                     # adapter in: SEQ -> CHF
        h = self.inner_drop(self.act(self.conv1(nn.functional.pad(h, (self.pad, 0)))))
        h = self.act(self.conv2(nn.functional.pad(h, (self.pad, 0))))
        return h.transpose(1, 2)                                           # adapter out: CHF -> SEQ
