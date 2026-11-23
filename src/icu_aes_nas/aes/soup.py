"""Model soup for one architecture (D18): pretrain once, fine-tune N ingredients, average weights.

Stages (each can run as its own SLURM job; all are resumable):
    pretrain     -> masked-value reconstruction on the train split
    linear_probe -> head-only training on the frozen pretrained encoder (LP of LP-FT); the shared starting point
    ingredient   -> fine-tune copy i from the linear-probe weights, head included (different seed and learning rate)
    combine    -> uniform soup, greedy soup (selection on meta), and the ensemble of the same ingredients
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from ..core.network import BlockNetwork, NetworkSpec
from ..models.experts import spec_from_cfg
from ..training.layout import all_eval_dirs, predictions_path
from ..training.predict import predict_deep, save_predictions
from ..training.train import load_pp, train_deep
from ..utils.io import atomic_write_json, read_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def soup_root(cfg) -> Path:
    return Path(cfg.paths.artifacts) / "soup" / cfg.task.name / f"split{cfg.split_seed}" / cfg.aes.soup.architecture


def _spec(cfg) -> NetworkSpec:
    arch_cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "model" / f"{cfg.aes.soup.architecture}.yaml")
    return spec_from_cfg(arch_cfg, load_pp(cfg), bool(cfg.trainer.use_missingness_mask))


def pretrain(cfg) -> dict:
    sc = cfg.aes.soup
    return train_deep(cfg, _spec(cfg), "soup_pretrain", 0, run_dir=soup_root(cfg) / "pretrain",
                      max_epochs=sc.pretrain_epochs, pretrain_mask_ratio=sc.mask_ratio,
                      write_predictions=False, method="soup")


def linear_probe(cfg) -> dict:
    """LP step of LP-FT (Kumar et al. 2022): fit only the classification head on the frozen pretrained encoder
    (train split, early stopping on val AUROC). Its weights, head included, are the common starting point of all
    ingredients, as model soups require (Wortsman et al. 2022). A random head per ingredient breaks averaging."""
    pre = read_json(soup_root(cfg) / "pretrain" / "done.json")
    state = torch.load(pre["net"], map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("head.")}   # pretraining head reconstructs values
    return train_deep(cfg, _spec(cfg), "soup_linear_probe", seed=0, run_dir=soup_root(cfg) / "linear_probe",
                      init_state=state, write_predictions=False, method="soup", trainable_prefixes=("head.",))


def ingredient(cfg, i: int) -> dict:
    lp = soup_root(cfg) / "linear_probe" / "done.json"
    if not lp.exists():
        raise FileNotFoundError(f"{lp} missing: run the soup stage `linear_probe` first")
    state = torch.load(read_json(lp)["net"], map_location="cpu", weights_only=True)   # identical init, head included
    rng = np.random.default_rng(i)
    cfg_i = copy.deepcopy(cfg)
    cfg_i.trainer.lr = float(cfg.trainer.lr * rng.choice([0.3, 0.5, 1.0, 2.0]))
    return train_deep(cfg_i, _spec(cfg), f"soup_ingredient{i}", seed=1000 + i,
                      run_dir=soup_root(cfg) / f"ingredient{i}", init_state=state, write_predictions=False,
                      method="soup")


def _meta_auroc(net, cfg, use_mask) -> float:
    dirs = [d for d in all_eval_dirs(cfg) if d[1] == "meta"]
    pred = predict_deep(net, dirs, use_mask=use_mask)
    return roc_auc_score(pred["y"], pred["logit"])


def combine(cfg) -> dict:
    root = soup_root(cfg)
    spec = _spec(cfg)
    use_mask = bool(cfg.trainer.use_missingness_mask)
    states = []
    for i in range(cfg.aes.soup.n_ingredients):
        done = root / f"ingredient{i}" / "done.json"
        if done.exists():
            states.append(torch.load(read_json(done)["net"], map_location="cpu", weights_only=True))
    if len(states) < 2:
        raise FileNotFoundError("Need at least two finished ingredients")

    def soup(sel):
        net = BlockNetwork(spec)
        avg = {k: sum(states[j][k].float() for j in sel) / len(sel) for k in states[0]}
        net.load_state_dict(avg)
        return net

    nets = [soup([j]) for j in range(len(states))]
    scores = [_meta_auroc(n, cfg, use_mask) for n in nets]
    order = list(np.argsort(scores)[::-1])
    greedy, best = [order[0]], scores[order[0]]
    for j in order[1:]:                       # greedy soup (Wortsman et al. 2022), selection on meta
        s = _meta_auroc(soup(greedy + [j]), cfg, use_mask)
        if s >= best:
            greedy, best = greedy + [j], s
    dirs = all_eval_dirs(cfg)
    for name, net in (("soup_uniform", soup(list(range(len(states))))), ("soup_greedy", soup(greedy)),
                      ("soup_best_single", nets[order[0]])):
        save_predictions(predict_deep(net, dirs, use_mask=use_mask), predictions_path(cfg, name, 0))
    # Fair comparison for the soup: ensemble of the SAME ingredients (probability average)
    preds = [predict_deep(n, dirs, use_mask=use_mask) for n in nets]
    ens = preds[0].copy()
    p = np.mean([1 / (1 + np.exp(-q["logit"].to_numpy())) for q in preds], axis=0).clip(1e-7, 1 - 1e-7)
    ens["logit"] = np.log(p / (1 - p))
    save_predictions(ens, predictions_path(cfg, "soup_ingredient_ensemble", 0))
    report = {"n_ingredients": len(states), "init": "linear_probe (LP-FT)", "meta_auroc_single": scores, "greedy_members": [int(g) for g in greedy],
              "meta_auroc_greedy": best}
    atomic_write_json(report, root / "report.json")
    return report
