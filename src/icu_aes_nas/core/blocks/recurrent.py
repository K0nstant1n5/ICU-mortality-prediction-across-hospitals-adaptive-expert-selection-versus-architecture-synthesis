import torch
from torch import nn

from ..bus import TokenBus
from .base import Block, register


@register("gru")
class GRUBlock(Block):
    """Unidirectional GRU layer: causal by construction. Native format SEQ."""

    def __init__(self, d: int, dropout: float = 0.1, **kwargs):
        super().__init__(d, dropout)
        self.gru = nn.GRU(d, d, batch_first=True)

    def mix(self, bus: TokenBus) -> torch.Tensor:
        # Padding is at the end (right-padded), so running through it never affects real tokens.
        out, _ = self.gru(bus.zero_padding().tokens)
        return out
