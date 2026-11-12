"""Metrics from proposal §12.1. All take labels y (0/1) and predicted probabilities p."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def brier(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def scaled_brier(y, p) -> float:
    ref = brier(y, np.full_like(p, y.mean()))
    return float(1 - brier(y, p) / ref) if ref > 0 else float("nan")


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def all_metrics(y, p, n_bins: int = 15) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "prevalence": float(y.mean())}
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": brier(y, p), "scaled_brier": scaled_brier(y, p), "ece": ece(y, p, n_bins),
        "cal_slope": slope, "cal_intercept": intercept,
    }
