"""Uncertainty and comparison statistics (proposal §12.2-§12.5)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# --- DeLong (1988) for two correlated AUROCs, fast version (Sun & Xu 2014) ----------------------------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, p1, p2) -> dict:
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff),
            "se": float(np.sqrt(max(var, 0))), "p": float(2 * stats.norm.sf(abs(z)))}


# --- bootstrap ------------------------------------------------------------------------------------------
def _resample_index(groups: np.ndarray, rng) -> np.ndarray:
    """Cluster bootstrap: resample groups (patients, or hospitals on eICU), keep all their rows."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inv), np.cumsum(np.bincount(inv))[:-1])
    pick = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([members[i] for i in pick])


def bootstrap_ci(metric, y, p, groups=None, n_boot=1000, ci=0.95, seed=0) -> tuple[float, float, float]:
    """(point, low, high). groups=None resamples rows; otherwise whole clusters."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    vals = []
    for _ in range(n_boot):
        idx = _resample_index(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap(metric, y, p1, p2, groups=None, n_boot=1000, ci=0.95, seed=0) -> dict:
    """CI of metric(p1) - metric(p2) on the same resampled units (§12.3)."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_index(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], p1[idx]) - metric(y[idx], p2[idx]))
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(metric(y, p1) - metric(y, p2)), "lo": float(lo), "hi": float(hi),
            "p_boot": float(min(1.0, p_two))}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (D15)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p))


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two discrete distributions (router choices, §12.7)."""
    a, b = np.asarray(a, float) + 1e-12, np.asarray(b, float) + 1e-12
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    return float(0.5 * (a * np.log2(a / m)).sum() + 0.5 * (b * np.log2(b / m)).sum())
