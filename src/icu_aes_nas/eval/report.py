"""Build the results tables: per method x evaluation set metrics with CIs, and the pre-registered comparisons.

Only call this after the method list is final (pre-registration, D15). Every call is appended to
`artifacts/reports/evaluation_log.jsonl` so looks at the test sets are on record.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..training.layout import predictions_path
from ..utils.io import append_jsonl, atomic_write_json, ensure_dir
from ..utils.logging import get_logger
from .metrics import all_metrics
from .stats import bootstrap_ci, delong_test, holm, paired_bootstrap

log = get_logger(__name__)

EVAL_SETS = ("test", "temporal", "external")


def _groups(df: pd.DataFrame, split: str, cfg) -> np.ndarray:
    col = (cfg.eval.cluster_column or {}).get(split)
    unit = df[col] if col and col in df else df["stay_id"]
    return (df["source"].astype(str) + ":" + unit.astype(str)).to_numpy()


def discover_methods(cfg) -> list[str]:
    root = Path(cfg.paths.artifacts) / "predictions" / cfg.task.name / f"split{cfg.split_seed}"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def load(cfg, method: str, seed=0) -> pd.DataFrame:
    return pd.read_parquet(predictions_path(cfg, method, seed))


def evaluate(cfg, methods: list[str] | None = None, seed=0) -> dict:
    methods = methods or discover_methods(cfg)
    out_dir = ensure_dir(Path(cfg.paths.artifacts) / "reports" / cfg.task.name / f"split{cfg.split_seed}")
    append_jsonl({"time": time.time(), "methods": methods, "seed": seed}, out_dir / "evaluation_log.jsonl")
    rows = []
    preds = {m: load(cfg, m, seed) for m in methods if predictions_path(cfg, m, seed).exists()}
    for m, df in preds.items():
        for (source, split), sub in df.groupby(["source", "split"]):
            if split not in EVAL_SETS:
                continue
            y, p = sub["y"].to_numpy(), sub["p"].to_numpy()
            r = {"method": m, "source": source, "split": split, **all_metrics(y, p, cfg.eval.ece_bins)}
            if "auroc" in r:
                _, r["auroc_lo"], r["auroc_hi"] = bootstrap_ci(
                    lambda a, b: roc_auc_score(a, b), y, p, _groups(sub, split, cfg), cfg.eval.n_bootstrap, cfg.eval.ci)
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"metrics_seed{seed}.csv", index=False)

    # best single expert on MIMIC validation decides what "best_expert" means (never chosen on test)
    alias = {}
    val_scores = {}
    for m, df in preds.items():
        v = df[df.split == "val"]
        if len(v) and v["y"].nunique() > 1 and not m.startswith(("aes_", "oracle", "nas_", "soup", "static_best",
                                                                     "mean_ensemble", "stacking", "knora")) and "_no" not in m:
            val_scores[m] = roc_auc_score(v["y"], v["p"])
    if val_scores:
        alias["best_expert"] = max(val_scores, key=val_scores.get)
    nas = [m for m in preds if m.startswith("nas_") and m.endswith("_best")]
    if nas:
        main = "nas_evolution_uniform_inheritTrue_best"
        alias["nas_best"] = main if main in nas else sorted(nas)[0]

    comps = []
    for a, b in cfg.eval.primary_comparisons:
        ma, mb = alias.get(a, a), alias.get(b, b)
        if ma not in preds or mb not in preds:
            continue
        key = ["source", "split", "stay_id", "time"]
        j = preds[ma].merge(preds[mb][key + ["p"]], on=key, suffixes=("_a", "_b"))
        for (source, split), sub in j.groupby(["source", "split"]):
            if split not in EVAL_SETS or sub["y"].nunique() < 2:
                continue
            y = sub["y"].to_numpy()
            dl = delong_test(y, sub["p_a"].to_numpy(), sub["p_b"].to_numpy())
            pb = paired_bootstrap(lambda a_, b_: roc_auc_score(a_, b_), y, sub["p_a"].to_numpy(),
                                  sub["p_b"].to_numpy(), _groups(sub, split, cfg), cfg.eval.n_bootstrap, cfg.eval.ci)
            comps.append({"a": ma, "b": mb, "source": source, "split": split, "auroc_diff": dl["diff"],
                          "diff_lo": pb["lo"], "diff_hi": pb["hi"], "p_delong": dl["p"]})
    if comps:
        adj = holm([c["p_delong"] for c in comps])
        for c, pa in zip(comps, adj):
            c["p_holm"] = pa
    pd.DataFrame(comps).to_csv(out_dir / f"comparisons_seed{seed}.csv", index=False)
    summary = {"aliases": alias, "n_methods": len(preds), "n_comparisons": len(comps)}
    atomic_write_json(summary, out_dir / f"summary_seed{seed}.json")
    log.info("Wrote %s", out_dir)
    return {"table": table, "comparisons": comps, **summary}
