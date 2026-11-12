"""Regularised evolution (Real et al. 2019) and random search with the same budget (D29, D30).

All state is written after every candidate, so a job killed by the 8 h limit loses at most one candidate.
The controller checks its time budget before each candidate and returns `incomplete` early; the CLI then
exits with code 75 and the sbatch wrapper resubmits it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..constants import DYNAMIC_VARS
from ..core.network import BlockNetwork
from ..data.dataset import make_loader
from ..training.layout import nas_dir, split_paths
from ..training.train import load_pp, train_deep
from ..utils.io import append_jsonl, atomic_write_json, ensure_dir, read_json, read_jsonl
from ..utils.logging import get_logger
from ..utils.slurm import seconds_left
from .genome import Genome
from .inherit import ExpertBank, fit_stitches, inherited_state
from .proxies import PROXIES
from .space import SearchSpace

log = get_logger(__name__)


def _rng_to_json(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def _rng_from_json(state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


class NASRunner:
    def __init__(self, cfg, family_prior: dict | None = None):
        self.cfg = cfg
        self.c = cfg.nas
        self.dir = ensure_dir(nas_dir(cfg))
        self.state_path = self.dir / "state.json"
        self.history_path = self.dir / "history.jsonl"
        self.pp = load_pp(cfg)
        model_root = Path(cfg.paths.artifacts) / "models" / cfg.task.name / f"split{cfg.split_seed}"
        self.bank = ExpertBank(model_root, list(self.c.inherit_from)) if self.c.inherit else None
        specs = self.bank.specs if self.bank else {}
        self.use_mask = bool(cfg.trainer.use_missingness_mask)
        if self.state_path.exists():
            st = read_json(self.state_path)
            rng = _rng_from_json(st["rng"])
        else:
            st = {"strategy": self.c.strategy, "population": [], "n_evaluated": 0, "rng": None}
            rng = np.random.default_rng([cfg.seed, 7])
        self.state = st
        self.space = SearchSpace(self.c, specs, self.pp.task_type, family_prior, rng)
        self.seen = {r["key"]: r for r in read_jsonl(self.history_path)}
        self._batch = None
        self._durations: list[float] = []

    # -- evaluation ------------------------------------------------------------------------------------
    def _probe_batch(self) -> dict:
        if self._batch is None:
            loader = make_loader(split_paths(self.cfg, "train"), 64, True, 0, self.use_mask, seed=0)
            self._batch = next(iter(loader))
        return self._batch

    def _build(self, g: Genome):
        spec = g.to_spec(len(DYNAMIC_VARS), self.pp.static_dim, self.use_mask, self.pp.task_type,
                         dropout=0.1)
        net = BlockNetwork(spec)
        init, mapping, n_stitch = None, {}, 0
        if self.bank and any(b.get("source", "scratch") != "scratch" for b in spec.blocks):
            init, mapping = inherited_state(spec, self.bank)
            net.load_state_dict(init, strict=False)
            n_stitch = fit_stitches(net, mapping, self.bank, self._probe_batch())
            init = net.state_dict()
        return spec, net, init, n_stitch

    def proxy(self, g: Genome) -> float:
        _, net, _, _ = self._build(g)
        return PROXIES["grad_norm"](net, self._probe_batch())

    def evaluate(self, g: Genome, op: str, parent: str | None, proxy: float | None) -> dict:
        key = g.key()
        if key in self.seen:
            return self.seen[key]
        t0 = time.time()
        spec, net, init, n_stitch = self._build(g)
        res = train_deep(self.cfg, spec, f"nas_{key}", seed=self.cfg.seed, run_dir=self.dir / "candidates" / key,
                         max_epochs=int(self.c.low_fidelity_epochs), init_state=init, write_predictions=False,
                         method=f"nas_{self.c.strategy}")
        score = res["best_score"]
        if res["monitor"] == "val_loss":
            score = -score
        rec = {"key": key, "genome": g.to_dict(), "graph": g.to_graph(), "score": score,
               "monitor": res["monitor"], "op": op, "parent": parent, "proxy": proxy, "n_stitch": n_stitch,
               "params": sum(p.numel() for p in net.parameters()), "wall_s": round(time.time() - t0, 1),
               "index": self.state["n_evaluated"]}
        append_jsonl(rec, self.history_path)
        self.seen[key] = rec
        self.state["n_evaluated"] += 1
        self._durations.append(rec["wall_s"])
        return rec

    def _save(self):
        self.state["rng"] = _rng_to_json(self.space.rng)
        atomic_write_json(self.state, self.state_path)

    def _time_ok(self) -> bool:
        need = 1.5 * (np.mean(self._durations) if self._durations else 600) + 300
        return seconds_left(float(self.c.time_budget_hours) * 3600) > need

    # -- strategies -------------------------------------------------------------------------------------
    def _child(self, parent: Genome) -> tuple[Genome, str, float | None]:
        if not self.c.proxy.enabled:
            child, op = self.space.mutate(parent)
            return child, op, None
        cands = [self.space.mutate(parent) for _ in range(int(self.c.proxy.offspring_per_step))]
        scored = [(self.proxy(c), c, op) for c, op in cands]
        p, child, op = max(scored, key=lambda t: t[0])
        return child, op, p

    def step(self) -> None:
        pop = self.state["population"]
        if self.c.strategy == "random" or len(pop) < int(self.c.population):
            g = self.space.sample()
            rec = self.evaluate(g, "sample", None, None)
        else:
            idx = self.space.rng.choice(len(pop), size=min(int(self.c.sample_size), len(pop)), replace=False)
            parent = max((pop[i] for i in idx), key=lambda r: r["score"])
            child, op, p = self._child(Genome.from_dict(parent["genome"]))
            rec = self.evaluate(child, op, parent["key"], p)
        if self.c.strategy == "evolution":
            pop.append({"key": rec["key"], "genome": rec["genome"], "score": rec["score"]})
            if len(pop) > int(self.c.population):
                pop.pop(0)                    # remove the OLDEST, not the worst (regularised evolution)
        self._save()

    def run(self) -> str:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        while self.state["n_evaluated"] < int(self.c.budget):
            if not self._time_ok():
                log.info("Time budget nearly spent after %d candidates; stopping for resubmission",
                         self.state["n_evaluated"])
                self._save()
                return "incomplete"
            self.step()
            log.info("[%s] %d/%d evaluated", self.c.strategy, self.state["n_evaluated"], int(self.c.budget))
        self._save()
        return "done"


def top_k(cfg, k: int) -> list[dict]:
    hist = read_jsonl(nas_dir(cfg) / "history.jsonl")
    return sorted(hist, key=lambda r: r["score"], reverse=True)[:k]


def summary(cfg) -> dict:
    hist = read_jsonl(nas_dir(cfg) / "history.jsonl")
    if not hist:
        return {}
    from scipy.stats import kendalltau
    with_proxy = [r for r in hist if r.get("proxy") is not None]
    tau = kendalltau([r["proxy"] for r in with_proxy], [r["score"] for r in with_proxy])[0] if len(with_proxy) > 3 else None
    return {"n": len(hist), "best": max(r["score"] for r in hist), "proxy_kendall_tau": tau,
            "inherit_share": float(np.mean([any(g["source"] != "scratch" for g in r["genome"]["slots"]) for r in hist]))}
