"""Mamba block (Gu & Dao 2023). Uses `mamba_ssm` CUDA kernels when installed, else a pure-PyTorch scan.

The fallback is a straightforward sequential selective scan. It is slow for long sequences but exact, and at
T <= 168 hourly steps it is fast enough. It also runs on CPU, which the smoke tests need.
"""

import torch
from torch import nn

from ..bus import TokenBus
from .base import Block, register

try:  # pragma: no cover - depends on the cluster
    from mamba_ssm import Mamba as _CudaMamba
except Exception:  # noqa: BLE001
    _CudaMamba = None


class SelectiveSSM(nn.Module):
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        di = expand * d
        self.di, self.d_state = di, d_state
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.dt_rank = max(1, d // 16)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)      # short causal depthwise conv
        u = nn.functional.silu(u)
        dt, Bm, Cm = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = nn.functional.softplus(self.dt_proj(dt))                  # [B, L, di]  input-dependent step
        A = -torch.exp(self.A_log.float())                             # [di, N]
        h = x.new_zeros(B, self.di, self.d_state, dtype=torch.float32)
        ys = []
        for t in range(L):                                             # h_t = exp(dt A) h_{t-1} + dt B x_t
            dA = torch.exp(dt[:, t].float().unsqueeze(-1) * A)
            dBx = dt[:, t].float().unsqueeze(-1) * Bm[:, t].float().unsqueeze(1) * u[:, t].float().unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].float().unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1).to(x.dtype) + u * self.D
        return self.out_proj(y * nn.functional.silu(z))


@register("mamba")
class MambaBlock(Block):
    """Causal selective scan. Adapter: zero the padding (there is no attention mask in a scan)."""

    can_fuse_modalities = True   # only if tokens are ordered by time (Pipeline A must sort the bus)

    def __init__(self, d: int, dropout: float = 0.1, d_state: int = 16, use_cuda_kernel: bool = True, **kwargs):
        super().__init__(d, dropout)
        if use_cuda_kernel and _CudaMamba is not None and torch.cuda.is_available():
            self.ssm = _CudaMamba(d_model=d, d_state=d_state)
        else:
            self.ssm = SelectiveSSM(d, d_state=d_state)

    def mix(self, bus: TokenBus) -> torch.Tensor:
        return self.ssm(bus.zero_padding().tokens)
