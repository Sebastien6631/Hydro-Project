from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projet_hydro.model.architectures.lightgbm.features import build_features, build_future_features


def make_full_df(n_hours=400):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=n_hours))
    debit = 10 + np.cumsum(rng.normal(0, 0.1, size=n_hours))
    return pd.DataFrame(
        {
            "debit_m3s": debit,
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul,
            "altitude_S1": 300.0,
        },
        index=index,
    )


EXUTOIRE = {"lat": 43.13, "lon": 0.92}
BV_PARAMS = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}


def test_build_features_nominal_shapes_and_no_cible_leak():
    df = make_full_df()

    X, y, Date = build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})

    assert len(X) == len(y) == len(Date) == 64  # vérifié par exécution directe (burn-in 336h)
    assert "debit_m3s" not in X.columns
    assert (y.values == df.loc[Date, "debit_m3s"].values).all()


def test_build_features_raises_when_cible_col_absent():
    df = make_full_df().drop(columns=["debit_m3s"])

    with pytest.raises(ValueError, match="debit_m3s"):
        build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})


def test_build_features_drops_q_fct_p_if_present():
    df = make_full_df()
    df["Q_fct_P"] = 1.0

    X, y, Date = build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})

    assert "Q_fct_P" not in X.columns


def test_build_features_passes_weather_lags_through():
    df = make_full_df()

    X, y, Date = build_features(
        df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={}, weather_lags=[6]
    )

    assert "precipitation_lag_6h_S1" in X.columns


def test_build_features_keeps_row_when_only_cible_is_nan():
    df = make_full_df()
    df.loc[df.index[-1], "debit_m3s"] = np.nan

    X, y, Date = build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})

    assert df.index[-1] in Date
    assert pd.isna(y.iloc[-1])


def test_build_future_features_keeps_tail_rows_even_with_nan_features():
    df = make_full_df()
    horizon = 8
    df.iloc[-horizon:, df.columns.get_loc("debit_m3s")] = float("nan")

    X, y, Date = build_future_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=horizon, transit_amont={})

    assert len(Date) >= horizon
    assert Date[-horizon:].equals(df.index[-horizon:])
