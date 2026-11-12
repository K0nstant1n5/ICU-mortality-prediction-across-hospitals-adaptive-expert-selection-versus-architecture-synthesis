"""Where everything lives under `artifacts/`. One place, so SLURM scripts and analysis agree."""

from __future__ import annotations

from pathlib import Path

from ..data.dataset import split_dir
from ..data.prepare import enabled_sources


def processed_seed_root(cfg) -> Path:
    return Path(cfg.paths.processed) / cfg.task.name / f"split{cfg.split_seed}"


def model_dir(cfg, name: str, seed: int) -> Path:
    return Path(cfg.paths.artifacts) / "models" / cfg.task.name / f"split{cfg.split_seed}" / name / f"seed{seed}"


def predictions_path(cfg, method: str, seed: int | str = 0) -> Path:
    return (Path(cfg.paths.artifacts) / "predictions" / cfg.task.name / f"split{cfg.split_seed}" / method
            / f"seed{seed}.parquet")


def nas_dir(cfg) -> Path:
    return Path(cfg.paths.artifacts) / "nas" / cfg.task.name / f"split{cfg.split_seed}" / cfg.nas.tag


def split_paths(cfg, split: str, role: str | None = "train") -> list[Path]:
    """Processed dirs of one split for all enabled sources with the given role (None = any role)."""
    return [split_dir(cfg.paths.processed, cfg.task.name, cfg.split_seed, s, split)
            for s, r in enabled_sources(cfg).items() if role is None or r == role]


def all_eval_dirs(cfg) -> list[tuple[str, str, Path]]:
    """Every (source, split, dir) that exists, for writing per-sample predictions."""
    out = []
    for s in enabled_sources(cfg):
        for split in ("train", "val", "meta", "test", "temporal", "external"):
            p = split_dir(cfg.paths.processed, cfg.task.name, cfg.split_seed, s, split)
            if (p / "index.parquet").exists():
                out.append((s, split, p))
    return out
