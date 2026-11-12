"""Temperature scaling (Guo et al. 2017), fitted on the meta split only and never refitted on test sets (D14)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _nll(t, z, y):
    p = 1 / (1 + np.exp(-np.clip(z / t, -30, 30)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(_nll, bounds=(0.05, 20.0), args=(logits, y), method="bounded")
    return float(res.x)
