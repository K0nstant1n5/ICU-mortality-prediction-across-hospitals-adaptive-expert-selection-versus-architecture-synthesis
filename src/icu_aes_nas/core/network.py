"""BlockNetwork: stem -> [stitch?] block -> ... -> head, all on the token bus.

Every expert and every NAS candidate is a BlockNetwork built from a `NetworkSpec`, which is what makes blocks
transferable between models (weight inheritance in the NAS) and lets Pipeline A reuse the same machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .blocks import build_block
from .bus import TokenBus
from .heads import Head
from .stems import STEMS, StaticConditioning


@dataclass
class NetworkSpec:
    d_model: int = 64
    stem: str = "linear"
    blocks: list[dict] = field(default_factory=list)   # each: {"family": ..., optional kwargs, "stitch": bool}
    head: str = "last"
    static_mode: str = "add"
    dropout: float = 0.1
    n_vars: int = 48
    static_dim: int = 8
    use_mask: bool = True
    task_type: str = "static"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkSpec":
        d = dict(d)
        d["blocks"] = [dict(b) for b in d.get("blocks", [])]
        return cls(**d)


class Stitch(nn.Module):
    """Linear map between two blocks' representation spaces (identity-initialised; ridge init in nas.inherit)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, bus: TokenBus) -> TokenBus:
        return bus.with_tokens(self.lin(bus.tokens)).zero_padding()


class BlockNetwork(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        d = spec.d_model
        self.stem = STEMS[spec.stem](spec.n_vars, d, use_mask=spec.use_mask)
        self.static = StaticConditioning(spec.static_dim, d, spec.static_mode)
        self.stitches = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for b in spec.blocks:
            kw = {k: v for k, v in b.items() if k not in ("family", "stitch", "source")}
            self.blocks.append(build_block(b["family"], d, spec.dropout, **kw))
            self.stitches.append(Stitch(d) if b.get("stitch") else nn.Identity())
        self.final_norm = nn.LayerNorm(d)
        self.head = Head(d, spec.head, spec.task_type)

    def embed(self, batch: dict) -> TokenBus:
        return self.static(self.stem(batch), batch["static"])

    def run_blocks(self, bus: TokenBus, upto: int | None = None) -> TokenBus:
        for i, (stitch, block) in enumerate(zip(self.stitches, self.blocks)):
            if upto is not None and i >= upto:
                break
            bus = block(stitch(bus))
        return bus

    def block_input(self, batch: dict, i: int) -> TokenBus:
        """Bus as it enters block i (after its stitch). Used to fit stitching layers."""
        bus = self.run_blocks(self.embed(batch), upto=i)
        return self.stitches[i](bus) if i < len(self.blocks) else bus

    def forward(self, batch: dict) -> torch.Tensor:
        bus = self.run_blocks(self.embed(batch))
        bus = bus.with_tokens(self.final_norm(bus.tokens))
        return self.head(bus, n_prefix=self.static.n_prefix)
