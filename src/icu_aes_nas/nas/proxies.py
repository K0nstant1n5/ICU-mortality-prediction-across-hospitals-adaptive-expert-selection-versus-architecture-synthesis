"""Zero-cost proxies (Abdelfattah et al. 2021) computed on one batch, before training.

They are only used to pre-rank offspring; their usefulness is checked afterwards with Kendall's tau against
trained validation scores (proposal §12.6). NASWOT is omitted on purpose: it assumes ReLU networks.
"""

from __future__ import annotations

import torch

from ..models.module import masked_targets


def _loss(net, batch):
    logits, y = masked_targets(batch, net(batch))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y.float())


def grad_norm(net: torch.nn.Module, batch: dict) -> float:
    net.zero_grad()
    _loss(net, batch).backward()
    s = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    net.zero_grad()
    return float(s)


def snip(net: torch.nn.Module, batch: dict) -> float:
    net.zero_grad()
    _loss(net, batch).backward()
    s = sum((p.grad * p).abs().sum().item() for p in net.parameters() if p.grad is not None)
    net.zero_grad()
    return float(s)


PROXIES = {"grad_norm": grad_norm, "snip": snip}
