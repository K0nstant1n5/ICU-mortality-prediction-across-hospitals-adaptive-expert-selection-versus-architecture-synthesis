"""Shared fixtures. Everything runs on fabricated data (icu_aes_nas.data.synthetic); no real records."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

SMOKE_OVERRIDES = [
    "trainer=smoke",
    "data.sources.hirid.enabled=true",      # exercise the HiRID toggle and pooled multi-source training
    "data.n_split_seeds=1",
    "aes.soup.n_ingredients=2",
    "aes.soup.pretrain_epochs=1",
    "aes.router.epochs=20",
    "nas.budget=6",
    "nas.population=3",
    "nas.sample_size=2",
    "nas.low_fidelity_epochs=1",
    "nas.top_k=2",
    "nas.final_seeds=[0]",
    "nas.d_model_choices=[16]",
    "eval.n_bootstrap=50",
]


def make_cfg(workspace: Path, *overrides: str):
    with initialize_config_module(config_module="icu_aes_nas.conf", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            f"paths.root={workspace}", f"paths.data={workspace}/data",
            f"paths.artifacts={workspace}/artifacts", *SMOKE_OVERRIDES, *overrides])
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(scope="session")
def cfg_factory():
    return make_cfg


def prepared_workspace(root: Path, task: str = "mortality24", n_patients: int = 120) -> Path:
    """Synthetic cohorts + prepared arrays for one task. Idempotent within a session."""
    from icu_aes_nas.data.prepare import prepare_task
    from icu_aes_nas.data.synthetic import make_synthetic

    marker = root / f".prepared_{task}"
    if not marker.exists():
        make_synthetic(root / "data", n_patients=n_patients, seed=0, sources=("miiv", "eicu", "hirid"),
                       tasks=(task,))
        prepare_task(make_cfg(root, f"task={task}"))
        marker.touch()
    return root
