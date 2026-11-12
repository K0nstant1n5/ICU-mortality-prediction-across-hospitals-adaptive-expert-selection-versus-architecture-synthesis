import numpy as np

from icu_aes_nas.eval.metrics import all_metrics, ece
from icu_aes_nas.eval.stats import bootstrap_ci, delong_test, holm


def test_delong_identical_models_has_p_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    res = delong_test(y, p, p)
    assert np.isclose(res["p"], 1.0) or np.isnan(res["p"])


def test_delong_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = y + rng.normal(0, 0.6, 600)
    bad = y + rng.normal(0, 3.0, 600)
    assert delong_test(y, good, bad)["p"] < 0.01


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert np.all(np.asarray(adj) <= 1) and np.all(np.asarray(adj) >= [0.01, 0.04, 0.03])
    assert np.isclose(adj[0], 0.03)


def test_metrics_and_ci_run():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.random(200) * 0.4, 0, 1)
    m = all_metrics(y, p)
    assert 0.5 < m["auroc"] <= 1 and 0 <= ece(y, p) <= 1
    lo, hi = bootstrap_ci(lambda yy, pp: all_metrics(yy, pp)["auroc"], y, p, None, n_boot=50, seed=0)[1:]
    assert lo <= m["auroc"] <= hi
