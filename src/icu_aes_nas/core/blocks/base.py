from __future__ import annotations

from typing import Callable

from torch import nn

from ..bus import TokenBus

BLOCK_REGISTRY: dict[str, type["Block"]] = {}


def register(family: str) -> Callable:
    def deco(cls):
        cls.family = family
        BLOCK_REGISTRY[family] = cls
        return cls
    return deco


class Block(nn.Module):
    """Residual block on the token bus. Subclasses implement `mix` in their native format."""

    family: str = "base"
    native_format: str = "SEQ"        # SEQ [B,L,d] | CHF [B,d,L] (channels-first) | VARGRAPH | GRID
    can_fuse_modalities: bool = False  # may this block mix tokens of different types (Pipeline A)?

    def __init__(self, d: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def mix(self, bus: TokenBus) -> "torch.Tensor":  # noqa: F821
        raise NotImplementedError

    def forward(self, bus: TokenBus) -> TokenBus:
        normed = bus.with_tokens(self.norm(bus.tokens))
        out = bus.tokens + self.drop(self.mix(normed))
        return bus.with_tokens(out).zero_padding()


def build_block(family: str, d: int, dropout: float, **kwargs) -> Block:
    if family not in BLOCK_REGISTRY:
        raise KeyError(f"Unknown block family '{family}'. Known: {sorted(BLOCK_REGISTRY)}")
    return BLOCK_REGISTRY[family](d=d, dropout=dropout, **kwargs)
