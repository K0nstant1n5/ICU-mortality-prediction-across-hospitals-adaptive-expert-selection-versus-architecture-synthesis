"""Retrain the top-k NAS candidates at full fidelity with new seeds and select on mean validation score.

This reduces selection optimism (proposal §12.9). The winner's predictions are written as `nas_<tag>_best`
so evaluation treats it exactly like an expert.
"""

from __future__ import annotations

import numpy as np

from ..constants import DYNAMIC_VARS
from ..training.layout import nas_dir, predictions_path
from ..training.train import load_pp, train_deep
from ..utils.io import atomic_write_json, read_json
from ..utils.logging import get_logger
from .genome import Genome
from .search import NASRunner, top_k

log = get_logger(__name__)


def finalize(cfg) -> dict:
    pp = load_pp(cfg)
    use_mask = bool(cfg.trainer.use_missingness_mask)
    runner = NASRunner(cfg)
    results = {}
    for rec in top_k(cfg, int(cfg.nas.top_k)):
        g = Genome.from_dict(rec["genome"])
        scores = []
        for s in cfg.nas.final_seeds:
            spec, _, init, _ = runner._build(g)
            res = train_deep(cfg, spec, f"nasfinal_{rec['key']}", seed=int(s),
                             run_dir=nas_dir(cfg) / "final" / rec["key"] / f"seed{s}", init_state=init,
                             write_predictions=False, method=f"nas_{cfg.nas.tag}_final")
            sc = res["best_score"] if res["monitor"] == "val_auroc" else -res["best_score"]
            scores.append(sc)
        results[rec["key"]] = {"low_fidelity": rec["score"], "final_mean": float(np.mean(scores)),
                               "final_sd": float(np.std(scores)), "genome": rec["genome"]}
    best = max(results, key=lambda k: results[k]["final_mean"])
    # Predictions of the winner for every final seed
    from ..core.network import BlockNetwork, NetworkSpec
    from ..training.layout import all_eval_dirs
    from ..training.predict import predict_deep, save_predictions
    import torch
    for s in cfg.nas.final_seeds:
        d = nas_dir(cfg) / "final" / best / f"seed{s}"
        net = BlockNetwork(NetworkSpec.from_dict(read_json(d / "spec.json")))
        net.load_state_dict(torch.load(d / "net.pt", map_location="cpu", weights_only=True))
        preds = predict_deep(net, all_eval_dirs(cfg), use_mask=use_mask)
        save_predictions(preds, predictions_path(cfg, f"nas_{cfg.nas.tag}_best", int(s)))
    out = {"best": best, "candidates": results, "n_vars": len(DYNAMIC_VARS), "task_type": pp.task_type}
    atomic_write_json(out, nas_dir(cfg) / "final.json")
    log.info("NAS winner %s: %.4f", best, results[best]["final_mean"])
    return out
