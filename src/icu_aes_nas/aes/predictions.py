"""Align per-sample predictions of several experts into one wide table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..training.layout import predictions_path

KEY = ["source", "split", "stay_id", "time"]


def load_wide(cfg, experts: list[str], seed: int | str = 0) -> pd.DataFrame:
    wide = None
    for e in experts:
        df = pd.read_parquet(predictions_path(cfg, e, seed))
        df = df[KEY + ["row", "y", "hospital_id", "logit"]].rename(columns={"logit": f"logit_{e}"})
        if wide is None:
            wide = df
        else:
            wide = wide.merge(df[KEY + [f"logit_{e}"]], on=KEY, how="inner")
    if wide is None:
        raise ValueError("No experts given")
    return wide.reset_index(drop=True)


def logits_matrix(wide: pd.DataFrame, experts: list[str]) -> np.ndarray:
    return wide[[f"logit_{e}" for e in experts]].to_numpy(dtype=np.float64)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
