"""Heads read the EHR tokens of the bus and output logits: [B] (static task) or [B, T] (hourly task)."""

from __future__ import annotations

import torch
from torch import nn

from .bus import TokenBus


class Head(nn.Module):
    def __init__(self, d: int, kind: str, task_type: str):
        super().__init__()
        if kind not in ("last", "mean", "attnpool"):
            raise ValueError(kind)
        self.kind, self.task_type = kind, task_type
        self.score = nn.Linear(d, 1) if kind == "attnpool" else None
        self.out = nn.Linear(d, 1)

    def forward(self, bus: TokenBus, n_prefix: int = 0) -> torch.Tensor:
        h = bus.tokens[:, n_prefix:]
        mask = bus.mask[:, n_prefix:].to(h.dtype).unsqueeze(-1)
        if self.task_type == "dynamic":
            # Hourly prediction: every head must stay causal, so pooled heads use cumulative pooling.
            if self.kind == "last":
                pooled = h
            elif self.kind == "mean":
                pooled = torch.cumsum(h * mask, 1) / torch.cumsum(mask, 1).clamp_min(1)
            else:
                w = torch.exp(self.score(h).clamp(-20, 20)) * mask
                pooled = torch.cumsum(w * h, 1) / torch.cumsum(w, 1).clamp_min(1e-6)
            return self.out(pooled).squeeze(-1)
        if self.kind == "last":
            last = (mask.squeeze(-1).sum(1) - 1).clamp_min(0).long()
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
        elif self.kind == "mean":
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            s = self.score(h).masked_fill(mask == 0, -1e4)
            pooled = (torch.softmax(s, 1) * h).sum(1)
        return self.out(pooled).squeeze(-1)
