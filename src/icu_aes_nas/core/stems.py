"""Stems turn model inputs into the token bus. Pipeline A adds ECG and CXR stems here."""

from __future__ import annotations

import torch
from torch import nn

from ..constants import TOKEN_EHR, TOKEN_STATIC
from .bus import TokenBus


class StaticConditioning(nn.Module):
    """Statics either added to every hour token (`add`) or prepended as one token (`token`)."""

    def __init__(self, static_dim: int, d: int, mode: str):
        super().__init__()
        if mode not in ("add", "token", "none"):
            raise ValueError(mode)
        self.mode = mode
        self.proj = nn.Linear(static_dim, d) if mode != "none" else None

    @property
    def n_prefix(self) -> int:
        return 1 if self.mode == "token" else 0

    def forward(self, bus: TokenBus, static: torch.Tensor) -> TokenBus:
        if self.mode == "none":
            return bus
        s = self.proj(static)
        if self.mode == "add":
            return bus.with_tokens(bus.tokens + s.unsqueeze(1)).zero_padding()
        B = s.shape[0]
        ones = torch.ones(B, 1, dtype=torch.bool, device=s.device)
        return TokenBus(
            tokens=torch.cat([s.unsqueeze(1), bus.tokens], 1),
            mask=torch.cat([ones, bus.mask], 1),
            time=torch.cat([torch.zeros(B, 1, device=s.device), bus.time], 1),
            type=torch.cat([torch.full((B, 1), TOKEN_STATIC, device=s.device, dtype=torch.long), bus.type], 1),
        )


def _ehr_bus(tokens: torch.Tensor, batch: dict) -> TokenBus:
    valid = batch["valid"].bool()
    return TokenBus(tokens=tokens, mask=valid, time=batch["time"],
                    type=torch.full(valid.shape, TOKEN_EHR, dtype=torch.long, device=valid.device)).zero_padding()


class LinearStem(nn.Module):
    """[x ; m] -> Linear -> one EHR token per hour."""

    family = "linear"

    def __init__(self, n_vars: int, d: int, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        self.proj = nn.Linear(2 * n_vars if use_mask else n_vars, d)

    def forward(self, batch: dict) -> TokenBus:
        x = torch.cat([batch["x"], batch["m"]], -1) if self.use_mask else batch["x"]
        return _ehr_bus(self.proj(x), batch)


class GRUDStem(nn.Module):
    """GRU-D (Che et al. 2018): input decay towards the mean and hidden-state decay driven by `delta`.

    x is already forward-filled and z-scored (train mean = 0), so
        x_hat = m * x + (1 - m) * (gamma_x * x_last + (1 - gamma_x) * 0) = x * (m + (1 - m) * gamma_x)
    """

    family = "grud"

    def __init__(self, n_vars: int, d: int, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        self.w_gx = nn.Parameter(torch.zeros(n_vars))      # diagonal input decay
        self.b_gx = nn.Parameter(torch.zeros(n_vars))
        self.gh = nn.Linear(n_vars, d)                      # hidden decay
        self.cell = nn.GRUCell(2 * n_vars, d)
        self.d = d

    def forward(self, batch: dict) -> TokenBus:
        x, m, delta = batch["x"], batch["m"], batch["delta"]
        if not self.use_mask:
            m = torch.ones_like(m)
        gamma_x = torch.exp(-torch.relu(self.w_gx * delta + self.b_gx))
        x_hat = x * (m + (1 - m) * gamma_x)
        gamma_h = torch.exp(-torch.relu(self.gh(delta)))
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d)
        outs = []
        for t in range(T):
            h = self.cell(torch.cat([x_hat[:, t], m[:, t]], -1), gamma_h[:, t] * h)
            outs.append(h)
        return _ehr_bus(torch.stack(outs, 1), batch)


STEMS = {"linear": LinearStem, "grud": GRUDStem}
