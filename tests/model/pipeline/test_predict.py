from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from previ_r2d2.model.architectures.stacking import meteo_cols
from previ_r2d2.model.pipeline.predict import predict_test_set


def test_predict_test_set_returns_aligned_filtered_arrays():
    rng = np.random.default_rng(0)
    n_seq = 30
    horizon = 2

    pred_lstm_test = rng.normal(0, 1, (n_seq, horizon)).astype(np.float32)
    y_test = rng.normal(0, 1, (n_seq, horizon)).astype(np.float32)
    lstm_idx_test = np.arange(n_seq)
    t_last_test = np.arange(n_seq)

    n_train = 100
    index = pd.date_range("2026-01-01", periods=n_train + n_seq, freq="1h")
    df_full_ctx = pd.DataFrame(
        {"debit_m3s": 10 + np.cumsum(rng.normal(0, 0.1, n_train + n_seq)), "precipitation_S1": rng.uniform(0, 5, n_train + n_seq)},
        index=index,
    )
    meteo_feature_cols = meteo_cols(df_full_ctx)

    pred_lgbm_all = np.full(n_train + n_seq, np.nan)
    pred_lgbm_all[:] = np.log1p(10.0)

    meta = Ridge(alpha=1.0)
    meta_n_feat = 2 * horizon + 4 + 2 + len(meteo_feature_cols)
    meta.fit(rng.normal(0, 1, (50, meta_n_feat)), rng.normal(0, 1, (50, horizon)))
    meta_scaler = StandardScaler().fit(rng.normal(0, 1, (50, meta_n_feat)))

    yt_v, pl_v, pt_v, stk_v, df_sub, q_now_v, pred_lgbm_multi_test, pred_lstm_test_out, pred_stacking_multi, valid = predict_test_set(
        pred_lstm_test, y_test, lstm_idx_test, t_last_test, n_train,
        pred_lgbm_all, df_full_ctx, meta, meta_scaler, horizon, meteo_feature_cols, df_full_ctx.iloc[n_train:],
    )

    assert yt_v.shape == pl_v.shape == pt_v.shape == stk_v.shape == q_now_v.shape
    assert len(df_sub) == len(yt_v)
    assert not np.isnan(stk_v).any()
    assert pred_lgbm_multi_test.shape == (n_seq, horizon)
    assert pred_lstm_test_out.shape == (n_seq, horizon)
    assert pred_stacking_multi.shape == (n_seq, horizon)
    assert valid.shape == (n_seq,)
    assert valid.dtype == bool
    assert valid.sum() == len(yt_v)
