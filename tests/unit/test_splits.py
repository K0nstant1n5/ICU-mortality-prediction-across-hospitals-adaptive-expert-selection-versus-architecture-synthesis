import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from icu_aes_nas.data.splits import check_split, make_source_split

CFG = OmegaConf.create({"split_fractions": {"train": 0.6, "val": 0.15, "meta": 0.1, "test": 0.15},
                        "temporal_holdout": {"source": "miiv", "min_year": 2017}})


def _meta(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"stay_id": np.arange(n), "patient_id": np.arange(n) // 2,
                         "est_year": rng.integers(2008, 2020, n), "hospital_id": 1})


def _labels(meta):
    return pd.Series((meta.stay_id % 7 == 0).astype(int).to_numpy(), index=meta.stay_id)


def test_train_source_split_has_no_leak_and_temporal_holdout():
    meta = _meta()
    s = make_source_split(meta, _labels(meta), "miiv", "train", CFG, seed=0)
    check_split(s)
    m = s
    first = m.groupby("patient_id").est_year.transform("min")
    assert (first[m.split == "temporal"] >= 2017).all()
    assert (first[m.split != "temporal"] < 2017).all()
    assert s.groupby("patient_id").split.nunique().max() == 1
    assert set(s.split) == {"train", "val", "meta", "test", "temporal"}


def test_external_source_is_all_external():
    meta = _meta()
    s = make_source_split(meta, _labels(meta), "eicu", "test", CFG, seed=0)
    assert set(s.split) == {"external"}


def test_split_is_deterministic_and_seed_dependent():
    meta = _meta()
    a = make_source_split(meta, _labels(meta), "miiv", "train", CFG, seed=1)
    b = make_source_split(meta, _labels(meta), "miiv", "train", CFG, seed=1)
    c = make_source_split(meta, _labels(meta), "miiv", "train", CFG, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.split.equals(c.split)
