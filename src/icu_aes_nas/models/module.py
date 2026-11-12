"""Lightning module for every deep model (experts, soup ingredients, NAS candidates)."""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ..core.network import BlockNetwork, NetworkSpec


def masked_targets(batch: dict, logits: torch.Tensor):
    """Flatten logits/labels to the positions that carry a label."""
    y = batch["y"]
    if logits.dim() == 2:                       # hourly task
        mask = batch["ymask"] & batch["valid"]
        return logits[mask], y[mask]
    return logits, y


class BinaryTask(L.LightningModule):
    def __init__(self, spec: dict, lr: float = 1e-3, weight_decay: float = 1e-4, pos_weight: float | None = None,
                 pretrain_mask_ratio: float | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.net = BlockNetwork(NetworkSpec.from_dict(spec))
        self.pretrain = pretrain_mask_ratio is not None
        if self.pretrain:   # masked-value reconstruction for the shared soup checkpoint (D18)
            self.recon = nn.Linear(self.net.spec.d_model, self.net.spec.n_vars)
        self._val = []

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch)

    # --- supervised -------------------------------------------------------------------------------------
    def _loss(self, batch):
        logits, y = masked_targets(batch, self(batch))
        pw = None if self.hparams.pos_weight is None else torch.tensor(self.hparams.pos_weight, device=y.device)
        return nn.functional.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pw), logits, y

    # --- self-supervised pretraining --------------------------------------------------------------------
    def _pretrain_loss(self, batch):
        hide = (torch.rand_like(batch["m"]) < self.hparams.pretrain_mask_ratio) & (batch["m"] > 0)
        b = dict(batch)
        b["x"] = batch["x"].masked_fill(hide, 0.0)
        b["m"] = batch["m"].masked_fill(hide, 0.0)
        bus = self.net.run_blocks(self.net.embed(b))
        n_prefix = self.net.static.n_prefix
        pred = self.recon(self.net.final_norm(bus.tokens[:, n_prefix:]))
        return ((pred - batch["x"]) ** 2)[hide].mean() if hide.any() else pred.sum() * 0

    def training_step(self, batch, _):
        loss = self._pretrain_loss(batch) if self.pretrain else self._loss(batch)[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, _):
        if self.pretrain:
            loss = self._pretrain_loss(batch)
            self.log("val_loss", loss, batch_size=batch["x"].shape[0])
            return
        loss, logits, y = self._loss(batch)
        self.log("val_loss", loss, batch_size=batch["x"].shape[0])
        self._val.append((logits.detach().float().cpu(), y.detach().cpu()))

    def on_validation_epoch_end(self):
        if self.pretrain or not self._val:
            return
        logits = torch.cat([v[0] for v in self._val]).numpy()
        y = torch.cat([v[1] for v in self._val]).numpy()
        self._val.clear()
        auroc = roc_auc_score(y, logits) if len(np.unique(y)) > 1 else 0.5
        self.log("val_auroc", float(auroc), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
