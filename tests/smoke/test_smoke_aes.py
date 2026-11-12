"""Smoke: experts -> ensembles, KNORA-E, soft/hard router, oracle; model soup; evaluation report.

Run alone with: pytest -m smoke_aes
"""
import pandas as pd
import pytest

from icu_aes_nas.aes import soup
from icu_aes_nas.aes.run import aes_dir, run_aes
from icu_aes_nas.eval.report import evaluate
from icu_aes_nas.training.layout import predictions_path
from icu_aes_nas.training.train import train_expert

from ..conftest import make_cfg, prepared_workspace

pytestmark = pytest.mark.smoke_aes
EXPERTS = ["lgbm", "grud", "tcn"]
AES_OVERRIDE = f"aes.experts=[{','.join(EXPERTS)}]"


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    ws = prepared_workspace(tmp_path_factory.mktemp("aes"), "mortality24")
    for m in EXPERTS:
        extra = "model.params.n_estimators=20" if m == "lgbm" else "model.spec.d_model=16"
        train_expert(make_cfg(ws, f"model={m}", extra))
    return ws


def test_aes_methods_write_predictions(ws):
    cfg = make_cfg(ws, AES_OVERRIDE)
    run_aes(cfg, seed=0)
    for method in ["static_best", "mean_ensemble", "stacking", "knora_e", "aes_soft", "aes_hard", "oracle"]:
        p = pd.read_parquet(predictions_path(cfg, method, 0))
        assert p.p.between(0, 1).all(), method
    assert (aes_dir(cfg) / "nas_prior_seed0.json").exists()
    w = pd.read_parquet(aes_dir(cfg) / "router_weights_seed0.parquet")
    assert len(w) > 0


def test_model_soup_stages(ws):
    cfg = make_cfg(ws, AES_OVERRIDE, "model.spec.d_model=16", "aes.soup.n_ingredients=2")
    soup.pretrain(cfg)
    for i in range(2):
        soup.ingredient(cfg, i)
    out = soup.combine(cfg)
    assert "soup_uniform" in str(out) or (soup.soup_root(cfg) / "report.json").exists()


def test_evaluation_report(ws):
    cfg = make_cfg(ws, AES_OVERRIDE)
    run_aes(cfg, seed=0)
    res = evaluate(cfg, seed=0)
    table = res["table"]
    assert {"method", "split"} <= set(table.columns)
    assert "aes_soft" in set(table.method)
