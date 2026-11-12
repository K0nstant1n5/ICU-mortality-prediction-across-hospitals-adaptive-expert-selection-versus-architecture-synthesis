"""Smoke: cross-family NAS (evolution + random baseline, inherit + scratch, uniform + AES prior), resume, finalize.

Run alone with: pytest -m smoke_nas
"""
import json

import pandas as pd
import pytest

from icu_aes_nas.aes.run import run_aes
from icu_aes_nas.nas.finalize import finalize
from icu_aes_nas.nas.search import NASRunner, summary
from icu_aes_nas.training.layout import nas_dir, predictions_path
from icu_aes_nas.training.train import train_expert
from icu_aes_nas.utils.io import read_json

from ..conftest import make_cfg, prepared_workspace

pytestmark = pytest.mark.smoke_nas
EXPERTS = ["grud", "tcn", "transformer"]
COMMON = ["model.spec.d_model=16", f"nas.inherit_from=[{','.join(EXPERTS)}]", f"aes.experts=[{','.join(EXPERTS)}]"]


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    ws = prepared_workspace(tmp_path_factory.mktemp("nas"), "mortality24")
    for m in EXPERTS:
        train_expert(make_cfg(ws, f"model={m}", "model.spec.d_model=16"))
    return ws


@pytest.mark.parametrize("strategy,inherit", [("evolution", "true"), ("random", "true"), ("evolution", "false")])
def test_search_runs_to_budget(ws, strategy, inherit):
    cfg = make_cfg(ws, *COMMON, f"nas.strategy={strategy}", f"nas.inherit={inherit}")
    assert NASRunner(cfg).run() == "done"
    hist = [json.loads(line) for line in (nas_dir(cfg) / "history.jsonl").read_text().splitlines()]
    assert len(hist) == cfg.nas.budget
    assert all("cost" in h or "wall_s" in json.dumps(h) for h in hist)
    s = summary(cfg)
    assert "n_evaluated" in s or len(s) > 0


def test_search_resumes_after_interruption(ws):
    cfg = make_cfg(ws, *COMMON, "nas.tag=resume_check", "nas.budget=4", "nas.time_budget_hours=0")  # stops at once
    assert NASRunner(cfg).run() == "incomplete"
    cfg = make_cfg(ws, *COMMON, "nas.tag=resume_check", "nas.budget=4")
    assert NASRunner(cfg).run() == "done"
    n = len((nas_dir(cfg) / "history.jsonl").read_text().splitlines())
    assert n == 4


def test_aes_prior_and_finalize(ws):
    cfg = make_cfg(ws, *COMMON)
    run_aes(cfg, seed=0)
    prior = read_json(ws / "artifacts" / "aes" / "mortality24" / "split0" / "nas_prior_seed0.json")
    cfg = make_cfg(ws, *COMMON, "nas.prior=aes")
    assert NASRunner(cfg, prior["family_frequency"]).run() == "done"
    out = finalize(cfg)
    assert out
    p = pd.read_parquet(predictions_path(cfg, f"nas_{cfg.nas.tag}_best", 0))
    assert p.p.between(0, 1).all()
