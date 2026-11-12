"""Smoke: synthetic cohorts -> prepare (MIMIC + HiRID pooled, eICU external) -> every base model -> predictions.

Run alone with: pytest -m smoke_base
"""
import pandas as pd
import pytest

from icu_aes_nas.training.layout import predictions_path
from icu_aes_nas.training.train import train_expert

from ..conftest import make_cfg, prepared_workspace

pytestmark = pytest.mark.smoke_base
MODELS = ["lgbm", "gru", "grud", "tcn", "transformer", "mamba"]


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return prepared_workspace(tmp_path_factory.mktemp("base"), "mortality24")


@pytest.mark.parametrize("model", MODELS)
def test_expert_trains_and_predicts_every_split(ws, model):
    cfg = make_cfg(ws, f"model={model}", "model.spec.d_model=16" if model != "lgbm" else "model.params.n_estimators=20")
    res = train_expert(cfg)
    assert res["name"] == model
    preds = pd.read_parquet(predictions_path(cfg, model, 0))
    assert {"train", "val", "meta", "test", "external"} <= set(preds.split)
    assert set(preds.source) == {"miiv", "eicu", "hirid"}
    assert preds.p.between(0, 1).all() and preds.stay_id.notna().all()
    # second call is a no-op (done.json) - this is what makes SLURM resubmission safe
    assert train_expert(cfg)["name"] == model


def test_dynamic_task_aki(tmp_path_factory):
    ws = prepared_workspace(tmp_path_factory.mktemp("aki"), "aki")
    cfg = make_cfg(ws, "task=aki", "model=tcn", "model.spec.d_model=16")
    train_expert(cfg)
    preds = pd.read_parquet(predictions_path(cfg, "tcn", 0))
    assert preds.time.nunique() > 1     # hourly predictions


def test_missingness_ablation_has_own_name(ws):
    cfg = make_cfg(ws, "model=gru", "model.spec.d_model=16", "trainer.use_missingness_mask=false")
    assert train_expert(cfg)["name"] == "gru_nomask"
