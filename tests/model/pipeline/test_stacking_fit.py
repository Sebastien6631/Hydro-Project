from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from previ_r2d2.model.pipeline.stacking_fit import fit_stacking


def make_fixture():
    rng = np.random.default_rng(0)
    n = 100
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    df_full = pd.DataFrame(
        {
            "debit_m3s": 10 + np.cumsum(rng.normal(0, 0.1, n)),
            "precipitation_S1": rng.uniform(0, 5, n),
        },
        index=index,
    )
    horizon = 2
    n_lstm = 20
    t_last = np.arange(50, 50 + n_lstm)
    oof_lgbm_multi = rng.normal(1, 0.1, (n_lstm, horizon))
    oof_lstm_vals = rng.normal(1, 0.1, (n_lstm, horizon))
    y_train = rng.normal(1, 0.1, (n_lstm, horizon))
    return df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon


def test_fit_stacking_ridge_default_saves_artifacts_and_predicts(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture()

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    assert isinstance(result["meta"], Ridge)
    assert (tmp_path / "meta.pkl").exists()
    assert (tmp_path / "meta_scaler.pkl").exists()
    assert result["q90_train"] == pytest.approx(np.percentile(y_train[~np.isnan(y_train)], 90))

    preds = result["meta"].predict(result["meta_scaler"].transform(result["meta_X"]))
    assert preds.shape == result["meta_y"].shape


def test_fit_stacking_lgbm_meta_type_uses_multioutput_regressor(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture()

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path, meta_type="lgbm",
    )

    assert isinstance(result["meta"], MultiOutputRegressor)
