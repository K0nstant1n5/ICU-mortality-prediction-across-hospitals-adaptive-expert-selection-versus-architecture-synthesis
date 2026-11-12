import torch
from torch import nn

from ..bus import TokenBus
from .base import Block, register


@register("ffn")
class FFNBlock(Block):
    """Position-wise feed-forward network (no token mixing)."""

    def __init__(self, d: int, dropout: float = 0.1, **kwargs):
        super().__init__(d, dropout)
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def mix(self, bus: TokenBus) -> torch.Tensor:
        return self.net(bus.tokens)


@register("identity")
class IdentityBlock(Block):
    """Empty slot in a NAS genome."""

    def __init__(self, d: int, dropout: float = 0.0, **kwargs):
        super().__init__(d, dropout)

    def forward(self, bus: TokenBus) -> TokenBus:
        return bus

    def mix(self, bus: TokenBus) -> torch.Tensor:  # pragma: no cover
        return torch.zeros_like(bus.tokens)
