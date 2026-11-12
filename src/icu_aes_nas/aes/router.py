"""Learned AES router (proposal §11.10).

    s = MLP(phi(x)); s_k = -inf for unavailable experts; w = softmax(s)
    soft: p = sum_k w_k p_k      hard: p = p_argmax(w)

phi = [meta-features, expert logits, expert entropies, (source one-hot)]. Fitted on the meta split only.
The availability mask is all ones in Pipeline B; Pipeline A passes which modalities exist per sample.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .predictions import sigmoid


def router_inputs(F: np.ndarray, Z: np.ndarray) -> np.ndarray:
    P = sigmoid(Z)
    ent = -(P * np.log(np.clip(P, 1e-7, 1)) + (1 - P) * np.log(np.clip(1 - P, 1e-7, 1)))
    return np.concatenate([F, Z, ent], 1).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, d_in: int, k: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


class Router:
    def __init__(self, mode: str = "soft", hidden: int = 32, epochs: int = 200, lr: float = 1e-2,
                 weight_decay: float = 1e-3, patience: int = 20, seed: int = 0):
        assert mode in ("soft", "hard")
        self.mode, self.hidden, self.epochs, self.lr = mode, hidden, epochs, lr
        self.wd, self.patience, self.seed = weight_decay, patience, seed

    def _scores(self, X, avail):
        s = self.model(torch.from_numpy(self.scaler.transform(X).astype(np.float32)))
        return s.masked_fill(~torch.from_numpy(avail), -1e9)

    def fit(self, X, Z, y, avail=None, groups=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, k = Z.shape
        avail = np.ones((n, k), bool) if avail is None else avail
        self.scaler = StandardScaler().fit(X)
        self.model = _MLP(X.shape[1], k, self.hidden)
        # 80/20 split of the meta set by group (stay) for early stopping
        g = groups if groups is not None else np.arange(n)
        ug = np.unique(g)
        va_g = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False))
        va = np.isin(g, list(va_g))
        tr = ~va
        Zt, yt = torch.from_numpy(Z.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
        P = torch.sigmoid(Zt)
        if self.mode == "hard":
            loss_k = -(yt[:, None] * torch.log(P.clamp_min(1e-7)) + (1 - yt[:, None]) * torch.log((1 - P).clamp_min(1e-7)))
            target = loss_k.masked_fill(~torch.from_numpy(avail), 1e9).argmin(1)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        trm, vam = torch.from_numpy(tr), torch.from_numpy(va)
        for _ in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            s = self._scores(X, avail)
            if self.mode == "soft":
                p = (torch.softmax(s, 1) * P).sum(1).clamp(1e-7, 1 - 1e-7)
                losses = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p))
            else:
                losses = nn.functional.cross_entropy(s, target, reduction="none")
            losses[trm].mean().backward()
            opt.step()
            v = float(losses[vam].mean().detach()) if va.any() else float(losses.mean().detach())
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k_: t.clone() for k_, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def weights(self, X, avail=None) -> np.ndarray:
        self.model.eval()
        avail = np.ones((len(X), self.model.net[-1].out_features), bool) if avail is None else avail
        return torch.softmax(self._scores(X, avail), 1).numpy()

    def predict(self, X, Z, avail=None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (combined probability, routing weights)."""
        w = self.weights(X, avail)
        P = sigmoid(Z)
        if self.mode == "soft":
            return (w * P).sum(1), w
        return P[np.arange(len(P)), w.argmax(1)], w
