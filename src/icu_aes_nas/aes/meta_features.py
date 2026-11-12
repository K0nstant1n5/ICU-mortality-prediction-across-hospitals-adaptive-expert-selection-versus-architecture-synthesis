"""Cheap per-sample meta-features for routers (proposal §11.10): missingness, density, length, age, hour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import load_arrays, split_dir

FEATURES = ["miss_rate", "meas_per_hour", "hours_seen", "age", "hour"]


def compute_meta_features(cfg, wide: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(wide), len(FEATURES)), dtype=np.float32)
    wide = wide.reset_index(drop=True)
    for (source, split), sub in wide.groupby(["source", "split"]):
        a = load_arrays(split_dir(cfg.paths.processed, cfg.task.name, cfg.split_seed, source, split))
        rows = sub["row"].to_numpy().astype(np.int64)
        t = sub["time"].to_numpy().astype(np.int64).clip(min=0)
        M = np.asarray(a["M"][np.unique(rows)])
        valid = np.asarray(a["valid"][np.unique(rows)])
        pos = {r: i for i, r in enumerate(np.unique(rows))}
        ii = np.array([pos[r] for r in rows])
        cum_meas = np.cumsum(M.sum(-1), axis=1)                    # [n, T]
        cum_hours = np.cumsum(valid, axis=1).clip(min=1)
        meas = cum_meas[ii, t]
        hours = cum_hours[ii, t]
        n_vars = M.shape[-1]
        out[sub.index, 0] = 1 - meas / (hours * n_vars)
        out[sub.index, 1] = meas / hours
        out[sub.index, 2] = hours
        out[sub.index, 3] = np.asarray(a["S"][rows, 0])            # scaled, binned age
        out[sub.index, 4] = t
    return out


def source_onehot(wide: pd.DataFrame, train_sources: list[str]) -> np.ndarray:
    """Which training database a row comes from; all zeros for external sources (unknown at test time).

    Only used when there is more than one training source (Pipeline B with HiRID/AUMCdb). External rows get
    zeros on purpose: the router may not use the identity of the test database.
    """
    if len(train_sources) < 2:
        return np.zeros((len(wide), 0), dtype=np.float32)
    return np.stack([(wide["source"] == s).to_numpy() & (wide["split"] != "external").to_numpy()
                     for s in train_sources], 1).astype(np.float32)
