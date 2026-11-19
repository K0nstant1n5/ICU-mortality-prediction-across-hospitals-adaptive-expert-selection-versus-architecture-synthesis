"""Mamba block (Gu & Dao 2023): one set of parameters, two ways to run the selective scan.

* On a CUDA tensor, with `mamba_ssm` installed, the scan uses its fused `selective_scan_fn` kernel.
* Otherwise (CPU, or no kernel) it uses a sequential pure-PyTorch scan: slow for long sequences but exact, and
  at T <= 168 hourly steps fast enough. The smoke tests need this path on CPU.

Both paths compute the same function from the same parameters, so the choice is made per forward call from the
input's device, and a checkpoint trained with the kernel loads and runs on CPU (and vice versa). This also keeps
NAS weight inheritance independent of where an expert was trained.
"""

import warnings

import torch
from torch import nn

from ..bus import TokenBus
from .base import Block, register

try:  # pragma: no cover - depends on the cluster
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # mamba_ssm 2.2.2 uses deprecated torch.cuda.amp decorators
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
except Exception:  # noqa: BLE001
    _cuda_scan = None

KERNEL_AVAILABLE = _cuda_scan is not None


class SelectiveSSM(nn.Module):
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, use_cuda_kernel: bool = True):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
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
        if self.use_cuda_kernel and _cuda_scan is not None and u.is_cuda:
            # kernel layout: u, delta, z [B, di, L]; B, C [B, N, L]; out = (scan + D*u) * silu(z)
            t = lambda a: a.float().transpose(1, 2).contiguous()      # noqa: E731
            y = _cuda_scan(t(u), t(dt), A.contiguous(), t(Bm), t(Cm), D=self.D.float(), z=t(z))
            return self.out_proj(y.transpose(1, 2).to(x.dtype))
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
        self.ssm = SelectiveSSM(d, d_state=d_state, use_cuda_kernel=use_cuda_kernel)

    def mix(self, bus: TokenBus) -> torch.Tensor:
        return self.ssm(bus.zero_padding().tokens)
