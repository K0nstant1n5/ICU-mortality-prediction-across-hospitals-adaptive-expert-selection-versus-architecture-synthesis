"""Non-learned and lightly learned combiners: static best, mean ensemble, stacking, KNORA-E, oracle."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .predictions import logit, sigmoid


def static_best(Z_meta, y_meta) -> int:
    """Index of the expert with the best AUROC on the meta split."""
    return int(np.argmax([roc_auc_score(y_meta, Z_meta[:, k]) for k in range(Z_meta.shape[1])]))


def mean_ensemble(Z) -> np.ndarray:
    return logit(sigmoid(Z).mean(1))


def fit_stacking(Z_meta, y_meta, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000).fit(Z_meta, y_meta)


def oracle(Z, y) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample best expert using the label: an upper bound for any router (proposal §11.10)."""
    P = sigmoid(Z)
    loss = -(y[:, None] * np.log(np.clip(P, 1e-7, 1)) + (1 - y[:, None]) * np.log(np.clip(1 - P, 1e-7, 1)))
    choice = loss.argmin(1)
    return Z[np.arange(len(Z)), choice], choice


class KnoraE:
    """KNORA-Eliminate (Ko et al. 2008) on standardised meta-features."""

    def __init__(self, k: int = 7):
        self.k = k

    def fit(self, F_meta, Z_meta, y_meta):
        self.scaler = StandardScaler().fit(F_meta)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(F_meta))
        thr = []
        for j in range(Z_meta.shape[1]):     # per-expert Youden threshold (classes are imbalanced)
            fpr, tpr, t = roc_curve(y_meta, Z_meta[:, j])
            thr.append(t[np.argmax(tpr - fpr)])
        self.correct = ((Z_meta >= np.array(thr)[None, :]).astype(int) == y_meta[:, None])
        return self

    def predict(self, F, Z) -> np.ndarray:
        _, nbrs = self.nn.kneighbors(self.scaler.transform(F))
        out = np.empty(len(F))
        P = sigmoid(Z)
        for i, nb in enumerate(nbrs):
            sel = None
            for k in range(self.k, 0, -1):
                ok = self.correct[nb[:k]].all(0)
                if ok.any():
                    sel = ok
                    break
            out[i] = P[i, sel].mean() if sel is not None else P[i].mean()
        return logit(out)
