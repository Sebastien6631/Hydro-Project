from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projet_hydro.model.architectures.stacking import (
    slice_lgbm_multistep,
    meteo_cols,
    kge_components,
    build_meteo_lag,
    build_meta_features,
    safe_slope,
    build_meta_features_live,
)


def test_slice_lgbm_multistep_valid_and_invalid_positions():
    oof_lgbm = np.arange(20, dtype=np.float32)
    t_last = np.array([5, 17])

    result = slice_lgbm_multistep(oof_lgbm, t_last, horizon=3)

    assert result[0].tolist() == [6.0, 7.0, 8.0]
    assert np.isnan(result[1]).all()


def test_meteo_cols_detects_weather_columns_excludes_lat_lon():
    df = pd.DataFrame({
        "precipitation_S1": [1], "temperature_S2": [1], "precipitation_S3": [1],
        "latitude_S1": [1], "longitude_S1": [1], "random_col": [1],
    })

    result = meteo_cols(df)

    assert result == ["precipitation_S1", "temperature_S2", "precipitation_S3"]


def test_kge_components_near_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true + 1e-4

    result = kge_components(y_true, y_pred)

    assert result["kge"] == pytest.approx(1.0, abs=1e-3)
    assert result["r"] == pytest.approx(1.0, abs=1e-3)


def test_build_meteo_lag_shifts_and_fills_zero():
    index = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"precipitation_S1": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    result = build_meteo_lag(df, ["precipitation_S1"], horizon=2)

    assert result.flatten().tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_build_meteo_lag_returns_zeros_when_no_meteo_cols_present():
    index = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"precipitation_S1": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    result = build_meteo_lag(df, ["nonexistent"], horizon=2)

    assert result.shape == (5, 1)
    assert (result == 0.0).all()


def test_build_meta_features_shapes_and_valid_mask_excludes_nan_rows():
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
    n_lstm = 10
    t_last = np.arange(50, 50 + n_lstm)
    oof_lgbm_multi = rng.normal(1, 0.1, (n_lstm, horizon))
    oof_lstm_vals = rng.normal(1, 0.1, (n_lstm, horizon))
    oof_lgbm_multi[3, 0] = np.nan  # une ligne invalide (NaN sur un pas LGB)
    y_train = rng.normal(1, 0.1, (n_lstm, horizon))

    meta_X, meta_y, valid = build_meta_features(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon, ["precipitation_S1"]
    )

    # n_features = 2*horizon (lgb+lstm) + 6 (divergence,q_now_log,sin,cos,crue_flag,trend) + 1 meteo = 11.
    assert meta_X.shape == (9, 11)
    assert meta_y.shape == (9, 2)
    assert valid.tolist() == [True, True, True, False, True, True, True, True, True, True]
    np.testing.assert_array_equal(meta_X[:, :horizon], oof_lgbm_multi[valid])


def test_safe_slope_returns_zero_with_fewer_than_two_valid_points():
    assert safe_slope(np.array([np.nan, 1.0, np.nan])) == 0.0


def test_safe_slope_computes_linear_regression_slope():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    result = safe_slope(x)

    assert result == pytest.approx(1.0)


def test_build_meta_features_live_matches_build_meta_features_on_one_sample():
    rng = np.random.default_rng(0)
    n = 200
    horizon = 3
    index = pd.date_range("2026-01-01", periods=n, freq="1D")
    df_full = pd.DataFrame({
        "debit_m3s": 10 + np.cumsum(rng.normal(0, 0.3, n)),
        "precipitation_S1": rng.uniform(0, 5, n),
    }, index=index)
    meteo_feature_cols = ["precipitation_S1"]

    t_last = np.array([150])
    pred_lgbm_multi = rng.normal(1, 0.1, (1, horizon))
    pred_lstm_vals = rng.normal(1, 0.1, (1, horizon))
    y_train = rng.normal(1, 0.1, (1, horizon))

    meta_X, _, _ = build_meta_features(df_full, pred_lgbm_multi, pred_lstm_vals, t_last, y_train, horizon, meteo_feature_cols)

    q90_train = np.percentile(y_train[~np.isnan(y_train)], 90)
    q_now_log = float(np.log1p(df_full["debit_m3s"].values[t_last[0]]))
    month = int(df_full.index.month.values[t_last[0]])
    trend_series = df_full["debit_m3s"].rolling(72, min_periods=12).apply(safe_slope, raw=True).fillna(0)
    trend = float(trend_series.values[t_last[0]])
    meteo_arr = build_meteo_lag(df_full, meteo_feature_cols, horizon)
    meteo_now = meteo_arr[t_last[0]]

    meta_X_live = build_meta_features_live(pred_lgbm_multi[0], pred_lstm_vals[0], q_now_log, month, q90_train, trend, meteo_now)

    np.testing.assert_allclose(meta_X, meta_X_live)
