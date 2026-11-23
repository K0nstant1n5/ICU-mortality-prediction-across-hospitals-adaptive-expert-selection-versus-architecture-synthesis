"""Training entry points with checkpoint/resume for 8-hour SLURM jobs.

Resume logic: if a run directory has `done.json`, the run is skipped. Otherwise the newest of `last.ckpt`
and Lightning's SLURM `hpc_ckpt_*.ckpt` is restored. On SIGUSR1 (sent by `#SBATCH --signal=USR1@300`)
Lightning saves an hpc checkpoint and requeues the job itself.
"""

from __future__ import annotations

import signal
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

from ..core.network import NetworkSpec
from ..data.preprocess import FittedPreprocessor
from ..data.dataset import make_loader
from ..models.experts import spec_from_cfg
from ..models.gbdt import fit_lgbm, tabular_for
from ..models.module import BinaryTask
from ..utils.cost import count_flops, count_parameters, track_run
from ..utils.io import atomic_write_json, ensure_dir, read_json
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from ..utils.slurm import in_slurm
from .layout import all_eval_dirs, model_dir, predictions_path, processed_seed_root, split_paths
from .predict import attach_ids, predict_deep, save_predictions

log = get_logger(__name__)
EXIT_INCOMPLETE = 75   # sbatch wrappers resubmit on this code


class Incomplete(RuntimeError):
    pass


def load_pp(cfg) -> FittedPreprocessor:
    return FittedPreprocessor.load(processed_seed_root(cfg) / "preprocessor.json")


def _resume_ckpt(run_dir: Path) -> str | None:
    cands = list(run_dir.glob("hpc_ckpt_*.ckpt")) + list(run_dir.glob("last.ckpt"))
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None


class _FrozenEval(L.Callback):
    """Keep frozen top-level submodules in eval mode (no dropout), so a linear probe sees fixed features.
    Lightning calls `module.train()` at every epoch start, so this re-applies it each epoch."""

    def __init__(self, frozen_children: list[str]):
        self.frozen_children = frozen_children

    def on_train_epoch_start(self, trainer, pl_module):
        for name in self.frozen_children:
            getattr(pl_module.net, name).eval()


def freeze_except(net: torch.nn.Module, trainable_prefixes: tuple[str, ...]) -> list[str]:
    """Set requires_grad=False on every parameter whose name does not start with one of `trainable_prefixes`.
    Returns the names of top-level children that are fully frozen (for `_FrozenEval`)."""
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith(trainable_prefixes)
    return [name for name, child in net.named_children()
            if not any(p.requires_grad for p in child.parameters())]


def _precision(cfg) -> str:
    return str(cfg.trainer.precision) if torch.cuda.is_available() else "32"


def train_deep(cfg, spec: NetworkSpec, name: str, seed: int, run_dir: Path | None = None,
               max_epochs: int | None = None, init_state: dict | None = None, pretrain_mask_ratio=None,
               write_predictions: bool = True, method: str = "expert",
               trainable_prefixes: tuple[str, ...] | None = None) -> dict:
    """Train one deep model. `trainable_prefixes` (e.g. ("head.",)) freezes every other parameter and keeps the
    frozen submodules in eval mode: a linear probe on fixed features (used for the soup's LP-FT initialisation)."""
    run_dir = ensure_dir(run_dir or model_dir(cfg, name, seed))
    if (run_dir / "done.json").exists():
        log.info("%s already done, skipping", run_dir)
        return read_json(run_dir / "done.json")
    seed_everything(seed)
    tr = cfg.trainer
    use_mask = bool(tr.use_missingness_mask)
    train_loader = make_loader(split_paths(cfg, "train"), tr.batch_size, True, tr.num_workers, use_mask, seed)
    val_loader = make_loader(split_paths(cfg, "val"), tr.batch_size, False, tr.num_workers, use_mask)
    module = BinaryTask(spec.to_dict(), lr=tr.lr, weight_decay=tr.weight_decay,
                        pretrain_mask_ratio=pretrain_mask_ratio)
    if init_state is not None:
        missing, unexpected = module.net.load_state_dict(init_state, strict=False)
        log.info("Initialised from inherited weights (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    extra_callbacks = []
    if trainable_prefixes:
        frozen = freeze_except(module.net, tuple(trainable_prefixes))
        extra_callbacks.append(_FrozenEval(frozen))
        n_train = sum(p.numel() for p in module.net.parameters() if p.requires_grad)
        log.info("Training only %s (%d parameters); frozen in eval mode: %s", trainable_prefixes, n_train, frozen)
    atomic_write_json(spec.to_dict(), run_dir / "spec.json")

    monitor, mode = ("val_loss", "min") if (pretrain_mask_ratio or spec.task_type == "dynamic") else ("val_auroc", "max")
    ckpt = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=monitor, mode=mode, save_last=True)
    early = EarlyStopping(monitor=monitor, mode=mode, patience=tr.patience)
    plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)] if in_slurm() else []
    epochs = max_epochs or tr.max_epochs
    trainer = L.Trainer(
        max_epochs=epochs, accelerator=tr.accelerator, devices=1, precision=_precision(cfg),
        callbacks=[ckpt, early, *extra_callbacks], logger=CSVLogger(run_dir, name="logs"), default_root_dir=run_dir,
        max_time={"hours": float(tr.max_time_hours)}, plugins=plugins, enable_progress_bar=False,
        enable_model_summary=False, deterministic=False, log_every_n_steps=10,
    )
    example = next(iter(val_loader))
    with track_run(cfg.paths.cost_ledger, method, f"{name}/seed{seed}", task=cfg.task.name,
                   split_seed=cfg.split_seed, params=count_parameters(module.net)) as rec:
        trainer.fit(module, train_loader, val_loader, ckpt_path=_resume_ckpt(run_dir))
        rec["epochs"] = trainer.current_epoch
        rec["flops_per_batch"] = count_flops(module.net, example)
    finished = early.stopped_epoch > 0 or trainer.current_epoch >= epochs
    if not finished:
        raise Incomplete(f"{run_dir} stopped by time budget at epoch {trainer.current_epoch}; resubmit to resume")

    best_path = ckpt.best_model_path or ckpt.last_model_path
    best = BinaryTask.load_from_checkpoint(best_path, map_location="cpu")
    torch.save(best.net.state_dict(), run_dir / "net.pt")
    result = {"name": name, "seed": seed, "best_ckpt": best_path, "net": str(run_dir / "net.pt"),
              "best_score": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
              "monitor": monitor}
    if write_predictions and not pretrain_mask_ratio:
        preds = predict_deep(best.net, all_eval_dirs(cfg), use_mask=use_mask)
        save_predictions(preds, predictions_path(cfg, name, seed))
    atomic_write_json(result, run_dir / "done.json")
    return result


def train_gbdt(cfg, seed: int) -> dict:
    name = cfg.model.name if cfg.model.use_count_features else f"{cfg.model.name}_nocounts"
    run_dir = ensure_dir(model_dir(cfg, name, seed))
    if (run_dir / "done.json").exists():
        return read_json(run_dir / "done.json")
    task_type = load_pp(cfg).task_type
    with track_run(cfg.paths.cost_ledger, "expert", f"{name}/seed{seed}", task=cfg.task.name,
                   split_seed=cfg.split_seed):
        clf = fit_lgbm(cfg.model, split_paths(cfg, "train"), split_paths(cfg, "val"), task_type, seed)
    clf.booster_.save_model(str(run_dir / "model.txt"))
    frames = []
    dirs = all_eval_dirs(cfg)
    import pandas as pd
    for source, split, path in dirs:
        X, y, rows, _ = tabular_for([path], task_type, cfg.model.use_count_features)
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        frames.append(pd.DataFrame({"source": source, "split": split, "row": rows[:, 0], "time": rows[:, 1],
                                    "y": y, "logit": np.log(p / (1 - p))}))
    save_predictions(attach_ids(pd.concat(frames, ignore_index=True), dirs), predictions_path(cfg, name, seed))
    result = {"name": name, "seed": seed, "model": str(run_dir / "model.txt"),
              "best_iteration": int(clf.best_iteration_ or 0)}
    atomic_write_json(result, run_dir / "done.json")
    return result


def train_expert(cfg) -> dict:
    if cfg.model.kind == "gbdt":
        return train_gbdt(cfg, cfg.seed)
    pp = load_pp(cfg)
    spec = spec_from_cfg(cfg.model, pp, bool(cfg.trainer.use_missingness_mask))
    name = cfg.model.name if cfg.trainer.use_missingness_mask else f"{cfg.model.name}_nomask"
    return train_deep(cfg, spec, name, cfg.seed)
