"""The token bus: the one tensor format every block reads and writes (proposal §3.1).

    tokens [B, L, d]  float   one row per token (an hour; Pipeline A adds ECG/CXR tokens)
    mask   [B, L]     bool    True = real token, False = padding or missing modality
    time   [B, L]     float   hours since ICU admission
    type   [B, L]     long    0 static, 1 EHR hour, 2 ECG, 3 CXR

Blocks never see each other's native formats. Each block converts bus -> native -> bus in its adapter.
Pipeline A only has to add modality stems that emit tokens with their own `time` and `type`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn


@dataclass
class TokenBus:
    tokens: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor
    type: torch.Tensor

    def with_tokens(self, tokens: torch.Tensor) -> "TokenBus":
        return replace(self, tokens=tokens)

    @property
    def d(self) -> int:
        return self.tokens.shape[-1]

    def zero_padding(self) -> "TokenBus":
        """Zero padded tokens. Required before conv / scan blocks, which have no notion of a mask."""
        return self.with_tokens(self.tokens * self.mask.unsqueeze(-1).to(self.tokens.dtype))


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of real time in hours (works for fractional ECG/CXR times in Pipeline A)."""

    def __init__(self, d: int, max_period: float = 10_000.0):
        super().__init__()
        half = d // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.d = d

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        ang = time.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if enc.shape[-1] < self.d:
            enc = nn.functional.pad(enc, (0, self.d - enc.shape[-1]))
        return enc


def causal_mask(length: int, device) -> torch.Tensor:
    """Boolean [L, L], True where attention is NOT allowed (future positions)."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
