"""Weight inheritance from trained experts and ridge-initialised stitching layers (proposal §11.12).

For an inherited slot, the k-th inherited block of family F from expert E takes the weights of E's k-th block
of family F (wrapping around). A stitching layer before it is initialised by ridge regression so that the
candidate's representation is mapped onto what the expert's block expects:
    W* = (Z_A^T Z_A + lambda I)^-1 Z_A^T Z_B
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from ..core.network import BlockNetwork, NetworkSpec
from ..utils.io import read_json


class ExpertBank:
    """Lazily loads trained expert networks (spec + weights) from `artifacts/models/...`."""

    def __init__(self, model_root: Path, names: list[str], seed: int = 0):
        self.specs: dict[str, dict] = {}
        self.paths: dict[str, Path] = {}
        self._nets: dict[str, BlockNetwork] = {}
        for n in names:
            d = Path(model_root) / n / f"seed{seed}"
            if (d / "done.json").exists() and (d / "spec.json").exists():
                self.specs[n] = read_json(d / "spec.json")
                self.paths[n] = d / "net.pt"

    def net(self, name: str) -> BlockNetwork:
        if name not in self._nets:
            net = BlockNetwork(NetworkSpec.from_dict(self.specs[name]))
            net.load_state_dict(torch.load(self.paths[name], map_location="cpu", weights_only=True))
            self._nets[name] = net.eval()
        return self._nets[name]


def _expert_block_index(spec: dict, family: str, k: int) -> int:
    idx = [i for i, b in enumerate(spec["blocks"]) if b["family"] == family]
    return idx[k % len(idx)]


def inherited_state(spec: NetworkSpec, bank: ExpertBank) -> tuple[dict, dict[int, tuple[str, int]]]:
    """Return a partial state dict for the candidate and a map candidate_block -> (expert, expert_block)."""
    state, mapping = {}, {}
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for i, b in enumerate(spec.blocks):
        src = b.get("source", "scratch")
        if src == "scratch":
            continue
        k = counters[(src, b["family"])]
        counters[(src, b["family"])] += 1
        j = _expert_block_index(bank.specs[src], b["family"], k)
        mapping[i] = (src, j)
        for name, t in bank.net(src).blocks[j].state_dict().items():
            state[f"blocks.{i}.{name}"] = t.clone()
    return state, mapping


@torch.no_grad()
def fit_stitches(net: BlockNetwork, mapping: dict[int, tuple[str, int]], bank: ExpertBank, batch: dict,
                 lam: float = 1e-2) -> int:
    """Ridge-initialise every stitching layer on one training batch. Returns how many were fitted."""
    net.eval()
    fitted = 0
    for i, (src, j) in mapping.items():
        stitch = net.stitches[i]
        if not hasattr(stitch, "lin"):
            continue
        before = net.run_blocks(net.embed(batch), upto=i)
        target = bank.net(src).block_input(batch, j)
        m = before.mask & target.mask
        za, zb = before.tokens[m].double(), target.tokens[m].double()
        za1 = torch.cat([za, torch.ones(len(za), 1, dtype=za.dtype)], 1)
        a = za1.T @ za1 + lam * torch.eye(za1.shape[1], dtype=za.dtype)
        w = torch.linalg.solve(a, za1.T @ zb)                  # [(d+1), d]
        stitch.lin.weight.copy_(w[:-1].T.float())
        stitch.lin.bias.copy_(w[-1].float())
        fitted += 1
    net.train()
    return fitted
