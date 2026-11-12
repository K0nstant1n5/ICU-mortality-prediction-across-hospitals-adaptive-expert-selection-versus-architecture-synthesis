"""Fit every combiner on the meta split and write their predictions in the expert prediction format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.prepare import enabled_sources
from ..training.layout import predictions_path
from ..training.predict import save_predictions
from ..utils.cost import track_run
from ..utils.io import atomic_write_json, ensure_dir
from ..utils.logging import get_logger
from .calibration import fit_temperature
from .meta_features import compute_meta_features, source_onehot
from .predictions import load_wide, logit, logits_matrix
from .router import Router, router_inputs
from .selectors import KnoraE, fit_stacking, mean_ensemble, oracle, static_best

log = get_logger(__name__)

# Which block family each expert contributes (used to turn router choices into a NAS prior).
EXPERT_FAMILY = {"grud": "gru", "gru": "gru", "tcn": "tcn", "transformer": "attn", "mamba": "mamba"}


def aes_dir(cfg) -> Path:
    return Path(cfg.paths.artifacts) / "aes" / cfg.task.name / f"split{cfg.split_seed}"


def available_experts(cfg, seed) -> list[str]:
    found = [e for e in cfg.aes.experts if predictions_path(cfg, e, seed).exists()]
    missing = sorted(set(cfg.aes.experts) - set(found))
    if missing:
        log.warning("No predictions for %s (seed %s); routing over %s", missing, seed, found)
    if len(found) < 2:
        raise FileNotFoundError("AES needs predictions from at least two experts.")
    return found


def run_aes(cfg, seed: int = 0) -> dict:
    experts = available_experts(cfg, seed)
    wide = load_wide(cfg, experts, seed)
    train_sources = [s for s, r in enabled_sources(cfg).items() if r == "train"]
    F = np.concatenate([compute_meta_features(cfg, wide), source_onehot(wide, train_sources)], 1)
    Z = logits_matrix(wide, experts)
    y = wide["y"].to_numpy().astype(int)
    meta = (wide["split"] == "meta").to_numpy()
    groups = (wide["source"] + ":" + wide["stay_id"].astype(str)).to_numpy()

    outputs: dict[str, np.ndarray] = {}
    with track_run(cfg.paths.cost_ledger, "aes", f"aes/seed{seed}", task=cfg.task.name, split_seed=cfg.split_seed):
        best = static_best(Z[meta], y[meta])
        outputs["static_best"] = Z[:, best]
        outputs["mean_ensemble"] = mean_ensemble(Z)
        stack = fit_stacking(Z[meta], y[meta], cfg.aes.stacking_C)
        outputs["stacking"] = stack.decision_function(Z)
        knora = KnoraE(cfg.aes.knora_k).fit(F[meta], Z[meta], y[meta])
        outputs["knora_e"] = knora.predict(F, Z)
        X = router_inputs(F, Z)
        rc = cfg.aes.router
        routers = {}
        for mode in ("soft", "hard"):
            r = Router(mode, rc.hidden, rc.epochs, rc.lr, rc.weight_decay, rc.patience, seed).fit(
                X[meta], Z[meta], y[meta], groups=groups[meta])
            p, w = r.predict(X, Z)
            outputs[f"aes_{mode}"] = logit(p)
            routers[mode] = w
        outputs["oracle"], oracle_choice = oracle(Z, y)

    report = {"experts": experts, "static_best": experts[best], "temperatures": {}}
    for name, z in outputs.items():
        t = 1.0 if name == "oracle" else fit_temperature(z[meta], y[meta])
        report["temperatures"][name] = t
        df = wide[["source", "split", "row", "stay_id", "hospital_id", "time", "y"]].copy()
        df["logit"] = z / t
        save_predictions(df, predictions_path(cfg, name, seed))

    # Router analysis (proposal §12.7): mean weights and hard-choice distribution per (source, split).
    out_dir = ensure_dir(aes_dir(cfg))
    w = routers["soft"]
    rw = pd.DataFrame(w, columns=experts)
    rw["choice_hard"] = np.array(experts)[routers["hard"].argmax(1)]
    rw["choice_oracle"] = np.array(experts)[oracle_choice]
    rw[["source", "split", "stay_id", "time"]] = wide[["source", "split", "stay_id", "time"]]
    rw.to_parquet(out_dir / f"router_weights_seed{seed}.parquet", index=False)
    report["mean_weights"] = rw.groupby(["source", "split"])[experts].mean().round(4).reset_index().to_dict("records")

    # NAS prior (D31): how often each block family's expert wins under the hard router on the meta split.
    fam = pd.Series([EXPERT_FAMILY.get(e) for e in rw.loc[meta, "choice_hard"]]).dropna()
    freq = fam.value_counts(normalize=True).to_dict()
    atomic_write_json({"family_frequency": freq, "source": "aes_hard_meta"}, out_dir / f"nas_prior_seed{seed}.json")
    atomic_write_json(report, out_dir / f"report_seed{seed}.json")
    log.info("AES done: static best = %s", experts[best])
    return report
