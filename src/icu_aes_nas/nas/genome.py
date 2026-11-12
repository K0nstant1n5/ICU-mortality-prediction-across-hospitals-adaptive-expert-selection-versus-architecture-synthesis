"""Architectures as typed graphs (proposal §4.2).

A genome is a stem, 1..K block slots and a head. Each slot gene is (family, source), where source is
"scratch" or the name of a trained expert whose block of that family is inherited. The genome converts to
(1) a NetworkSpec that BlockNetwork can build and (2) a typed graph (nodes = block families, edges = tensor
formats) used for validity checks, logging and a later GNN surrogate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from ..core.blocks import BLOCK_REGISTRY
from ..core.network import NetworkSpec

MIXING_FAMILIES = {"gru", "tcn", "attn", "mamba"}   # blocks that mix information over time


@dataclass
class Gene:
    family: str
    source: str = "scratch"          # "scratch" or expert name
    kwargs: dict = field(default_factory=dict)


@dataclass
class Genome:
    stem: str
    slots: list[Gene]
    head: str
    d_model: int
    static_mode: str = "add"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(stem=d["stem"], slots=[Gene(**g) for g in d["slots"]], head=d["head"],
                   d_model=d["d_model"], static_mode=d.get("static_mode", "add"))

    def key(self) -> str:
        """Canonical hash (identity slots removed) so duplicates are not trained twice."""
        d = self.to_dict()
        d["slots"] = [g for g in d["slots"] if g["family"] != "identity"]
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_spec(self, n_vars: int, static_dim: int, use_mask: bool, task_type: str, dropout: float = 0.1) -> NetworkSpec:
        blocks = []
        prev_source = "stem"
        for g in self.slots:
            if g.family == "identity":
                continue
            b = {"family": g.family, "source": g.source, **g.kwargs}
            # A stitching layer is inserted where an inherited block's input comes from a different model.
            b["stitch"] = g.source != "scratch" and g.source != prev_source
            blocks.append(b)
            prev_source = g.source
        return NetworkSpec(d_model=self.d_model, stem=self.stem, blocks=blocks, head=self.head,
                           static_mode=self.static_mode, dropout=dropout, n_vars=n_vars, static_dim=static_dim,
                           use_mask=use_mask, task_type=task_type)

    def to_graph(self) -> dict:
        """Typed DAG: node types = families, edge types = tensor formats entering the target node."""
        nodes = [{"id": 0, "type": f"stem:{self.stem}"}]
        edges = []
        for g in self.slots:
            if g.family == "identity":
                continue
            nid = len(nodes)
            nodes.append({"id": nid, "type": g.family, "source": g.source})
            fmt = BLOCK_REGISTRY[g.family].native_format
            edges.append({"src": nid - 1, "dst": nid, "format": fmt})
        nodes.append({"id": len(nodes), "type": f"head:{self.head}"})
        edges.append({"src": len(nodes) - 2, "dst": len(nodes) - 1, "format": "SEQ"})
        return {"nodes": nodes, "edges": edges}


def is_legal(g: Genome, expert_specs: dict[str, dict], task_type: str) -> tuple[bool, str]:
    """Type rules from the proposal (§3.3, §4.1)."""
    fams = [s.family for s in g.slots if s.family != "identity"]
    if g.stem != "grud" and not any(f in MIXING_FAMILIES for f in fams):
        return False, "no block mixes information over time"
    for s in g.slots:
        if s.family not in BLOCK_REGISTRY:
            return False, f"unknown family {s.family}"
        if s.source != "scratch":
            spec = expert_specs.get(s.source)
            if spec is None:
                return False, f"expert {s.source} not trained"
            if spec["d_model"] != g.d_model:
                return False, "inherited block width differs from d_model"
            if not any(b["family"] == s.family for b in spec["blocks"]):
                return False, f"expert {s.source} has no {s.family} block"
    if g.static_mode == "token" and task_type == "dynamic" and g.head != "last":
        return False, "token statics with pooled hourly heads are not supported"
    return True, "ok"
