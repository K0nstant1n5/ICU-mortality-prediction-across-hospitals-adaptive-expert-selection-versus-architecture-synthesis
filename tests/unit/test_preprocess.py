import numpy as np
import pandas as pd

from icu_aes_nas.constants import DYNAMIC_VARS
from icu_aes_nas.data.preprocess import FittedPreprocessor, _dense, _transform_dynamic


def _pp(seq_len=6):
    return FittedPreprocessor(dyn_mean={v: 0.0 for v in DYNAMIC_VARS}, dyn_std={v: 1.0 for v in DYNAMIC_VARS},
                              sta_mean={}, sta_std={}, bin_width=5, delta_cap=4, seq_len=seq_len,
                              task_type="static")


def test_forward_fill_mask_and_delta():
    x = np.full((1, 6, len(DYNAMIC_VARS)), np.nan, dtype=np.float32)
    x[0, 1, 0] = 2.0
    x[0, 4, 0] = 5.0
    f, m, d = _transform_dynamic(x, _pp())
    assert f[0, :, 0].tolist() == [0.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    assert m[0, :, 0].tolist() == [0, 1, 0, 0, 1, 0]
    # hours since last measurement / cap(4): never seen at hour 0 -> 0, then 0,1,2 after hour 1, capped
    assert np.allclose(d[0, :, 0].astype(float), [0, 0, 0.25, 0.5, 0, 0.25])
    assert (f[0, :, 1] == 0).all() and (m[0, :, 1] == 0).all()


def test_dense_valid_mask_right_pads_short_stays():
    dyn = pd.DataFrame({"stay_id": [1, 1, 2], "time": [0, 2, 0]})
    for v in DYNAMIC_VARS:
        dyn[v] = 1.0
    x, valid = _dense(dyn, [1, 2], 4)
    assert valid.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]
    assert np.isnan(x[0, 1]).all() and not np.isnan(x[0, 2]).any()
