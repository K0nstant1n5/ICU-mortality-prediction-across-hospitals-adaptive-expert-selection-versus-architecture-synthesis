"""Sampling and mutation over the genome space, with a family prior (uniform or AES-guided, D31)."""

from __future__ import annotations

import copy

import numpy as np

from .genome import Gene, Genome, is_legal

TCN_DILATIONS = [1, 2, 4, 8]


class SearchSpace:
    def __init__(self, nas_cfg, expert_specs: dict[str, dict], task_type: str, family_prior: dict | None = None,
                 rng: np.random.Generator | None = None):
        self.c = nas_cfg
        self.experts = expert_specs
        self.task_type = task_type
        self.rng = rng or np.random.default_rng(0)
        fams = list(nas_cfg.families)
        w = np.array([float((family_prior or {}).get(f, 0.0)) for f in fams])
        uniform = np.ones(len(fams)) / len(fams)
        # Mix the prior with uniform so every family stays reachable.
        self.family_p = uniform if w.sum() == 0 else 0.5 * uniform + 0.5 * w / w.sum()
        self.families = fams

    # -- helpers -----------------------------------------------------------------------------------------
    def _family(self) -> str:
        return str(self.rng.choice(self.families, p=self.family_p))

    def _source(self, family: str, d_model: int) -> str:
        if not self.c.inherit or self.rng.random() < 0.5:
            return "scratch"
        cands = [e for e, s in self.experts.items()
                 if e in self.c.inherit_from and s["d_model"] == d_model and any(b["family"] == family for b in s["blocks"])]
        return str(self.rng.choice(cands)) if cands else "scratch"

    def _gene(self, d_model: int) -> Gene:
        f = self._family()
        kw = {"dilation": int(self.rng.choice(TCN_DILATIONS))} if f == "tcn" else {}
        return Gene(f, self._source(f, d_model), kw)

    def legal(self, g: Genome) -> bool:
        return is_legal(g, self.experts, self.task_type)[0]

    # -- public ------------------------------------------------------------------------------------------
    def sample(self, max_tries: int = 200) -> Genome:
        for _ in range(max_tries):
            d = int(self.rng.choice(list(self.c.d_model_choices)))
            k = int(self.rng.integers(self.c.min_slots, self.c.max_slots + 1))
            g = Genome(stem=str(self.rng.choice(list(self.c.stems))), slots=[self._gene(d) for _ in range(k)],
                       head=str(self.rng.choice(list(self.c.heads))), d_model=d)
            if self.legal(g):
                return g
        raise RuntimeError("Could not sample a legal genome; check the search space config")

    def mutate(self, parent: Genome, max_tries: int = 100) -> tuple[Genome, str]:
        for _ in range(max_tries):
            g = copy.deepcopy(parent)
            op = str(self.rng.choice(["swap_family", "toggle_source", "insert", "delete", "head", "stem", "width"]))
            i = int(self.rng.integers(len(g.slots)))
            if op == "swap_family":
                g.slots[i] = self._gene(g.d_model)
            elif op == "toggle_source":
                s = g.slots[i]
                s.source = "scratch" if s.source != "scratch" else self._source(s.family, g.d_model)
            elif op == "insert" and len(g.slots) < self.c.max_slots:
                g.slots.insert(i, self._gene(g.d_model))
            elif op == "delete" and len(g.slots) > self.c.min_slots:
                g.slots.pop(i)
            elif op == "head":
                g.head = str(self.rng.choice(list(self.c.heads)))
            elif op == "stem":
                g.stem = str(self.rng.choice(list(self.c.stems)))
            elif op == "width" and all(s.source == "scratch" for s in g.slots):
                g.d_model = int(self.rng.choice(list(self.c.d_model_choices)))
            else:
                continue
            if self.legal(g) and g.key() != parent.key():
                return g, op
        return self.sample(), "resample"
