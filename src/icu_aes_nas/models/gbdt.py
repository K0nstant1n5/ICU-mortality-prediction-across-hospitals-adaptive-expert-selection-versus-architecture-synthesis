"""LightGBM expert on YAIB-style cumulative tabular features."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
from omegaconf import OmegaConf

from ..data.dataset import load_arrays
from ..data.features import build_tabular


def tabular_for(paths: list[Path], task_type: str, use_counts: bool):
    Xs, ys, rows, srcs = [], [], [], []
    for p in paths:
        if not (Path(p) / "index.parquet").exists():
            continue
        a = load_arrays(p)
        X, y, r = build_tabular(a, task_type, use_counts)
        Xs.append(X), ys.append(y), rows.append(r), srcs.append(np.full(len(y), str(p)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rows), np.concatenate(srcs)


def fit_lgbm(model_cfg, train_paths, val_paths, task_type: str, seed: int):
    params = OmegaConf.to_container(model_cfg.params, resolve=True)
    Xtr, ytr, _, _ = tabular_for(train_paths, task_type, model_cfg.use_count_features)
    Xva, yva, _, _ = tabular_for(val_paths, task_type, model_cfg.use_count_features)
    clf = lgb.LGBMClassifier(random_state=seed, **params)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(model_cfg.early_stopping_rounds, verbose=False)])
    return clf
