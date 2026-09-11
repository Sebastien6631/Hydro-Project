from __future__ import annotations

import numpy as np
import pandas as pd

from projet_hydro.model.features.amont import compute_amont_features


def test_compute_amont_features_returns_unchanged_df_when_no_amont_columns():
    df = pd.DataFrame({"debit_m3s": [1.0, 2.0, 3.0]})

    result = compute_amont_features(df, horizon=1, steps_per_day=1, transit_amont={})

    pd.testing.assert_frame_equal(result, df)


def test_compute_amont_features_seasonal_shift_uses_correct_shift_per_row():
    # 14 jours : positions 0-3 en février (hiver), positions 4-13 en mars (printemps).
    index = pd.date_range("2026-02-25", periods=14, freq="1D")
    values = pd.Series(np.arange(14, dtype=float), index=index)
    df = pd.DataFrame({"debit_amont": values, "debit_lag_1h": pd.Series(4.0, index=index)})
    transit_amont = {"debit_amont": {"hiver": 3, "printemps": 5, "ete": 1, "automne": 1}}

    result = compute_amont_features(df, horizon=1, steps_per_day=1, transit_amont=transit_amont)

    # hiver (positions 0-3) : shift_h = max(1,3) = 3 -> valeur = position - 3 (NaN si < 3)
    assert result["debit_amont"].iloc[0:3].isna().all()
    assert result["debit_amont"].iloc[3] == 0.0
    # printemps (positions 4-13) : shift_h = max(1,5) = 5 -> valeur = position - 5 (NaN si < 5)
    assert result["debit_amont"].iloc[4].__class__ is not None and pd.isna(result["debit_amont"].iloc[4])
    for pos in range(5, 14):
        assert result["debit_amont"].iloc[pos] == pos - 5


def test_compute_amont_features_seasonal_shift_floors_at_horizon_when_transit_is_smaller():
    # horizon=5 > transit_h=2 pour toutes les saisons -> shift_h doit rester à
    # l'horizon (5), pas tomber à transit_h (2). Toutes les dates sont en
    # janvier (hiver) pour isoler ce seul cas.
    index = pd.date_range("2026-01-10", periods=10, freq="1D")
    values = pd.Series(np.arange(10, dtype=float), index=index)
    df = pd.DataFrame({"debit_amont": values, "debit_lag_5h": pd.Series(4.0, index=index)})
    transit_amont = {"debit_amont": {"hiver": 2, "printemps": 2, "ete": 2, "automne": 2}}

    result = compute_amont_features(df, horizon=5, steps_per_day=1, transit_amont=transit_amont)

    expected = values.shift(5)
    pd.testing.assert_series_equal(result["debit_amont"], expected, check_names=False)


def test_compute_amont_features_falls_back_to_simple_shift_when_no_transit_config():
    index = pd.date_range("2026-01-01", periods=10, freq="1D")
    values = pd.Series(np.arange(10, dtype=float), index=index)
    df = pd.DataFrame({"debit_amont": values, "debit_lag_2h": pd.Series(4.0, index=index)})

    result = compute_amont_features(df, horizon=2, steps_per_day=1, transit_amont={})

    expected = values.shift(2)
    pd.testing.assert_series_equal(result["debit_amont"], expected, check_names=False)


def test_compute_amont_features_adds_lag_columns():
    index = pd.date_range("2026-01-01", periods=60, freq="1h")
    values = pd.Series(np.arange(60, dtype=float), index=index)
    df = pd.DataFrame({"debit_amont": values, "debit_lag_8h": pd.Series(4.0, index=index)})

    result = compute_amont_features(df, horizon=8, steps_per_day=24, transit_amont={})

    # transit_amont={} -> shift simple(8) : debit_amont[i] = i-8 (i>=8)
    for lag in [8, 20, 24, 32]:
        assert f"debit_amont_lag{lag}h" in result.columns
    for pos in range(32, 60):
        assert result["debit_amont_lag8h"].iloc[pos] == pos - 8
        assert result["debit_amont_lag20h"].iloc[pos] == pos - 20
        assert result["debit_amont_lag24h"].iloc[pos] == pos - 24
        assert result["debit_amont_lag32h"].iloc[pos] == pos - 32


def test_compute_amont_features_gradient_and_ratio_match_hand_computed_values():
    index = pd.date_range("2026-01-01", periods=60, freq="1h")
    values = pd.Series(np.arange(60, dtype=float), index=index)
    df = pd.DataFrame({"debit_amont": values, "debit_lag_8h": pd.Series(4.0, index=index)})

    result = compute_amont_features(df, horizon=8, steps_per_day=24, transit_amont={})

    # debit_amont_lag8h[i] = i-8 (i>=8, transit_amont={} -> shift simple).
    # gradient_6h = lag8h - amont.shift(6) -> pente constante d'une rampe = 6, valide i>=14.
    for pos in range(14, 60):
        assert result["debit_amont_gradient_6h"].iloc[pos] == 6.0
    # gradient_24h = lag8h - amont.shift(24) -> pente constante = 24, valide i>=32.
    for pos in range(32, 60):
        assert result["debit_amont_gradient_24h"].iloc[pos] == 24.0
    # ratio_aval = lag8h / (debit_lag_8h + EPS) = (i-8) / 4.000001, valide i>=8.
    for pos in [10, 30, 59]:
        expected = (pos - 8) / (4.0 + 1e-6)
        assert result["debit_amont_ratio_aval"].iloc[pos] == expected


def test_compute_amont_features_handles_multiple_amont_columns_independently():
    index = pd.date_range("2026-01-01", periods=40, freq="1h")
    values_a = pd.Series(np.arange(40, dtype=float), index=index)
    values_b = pd.Series(np.arange(40, dtype=float) + 1000.0, index=index)
    df = pd.DataFrame({
        "debit_amont_A": values_a,
        "debit_amont_B": values_b,
        "debit_lag_8h": pd.Series(4.0, index=index),
    })

    result = compute_amont_features(df, horizon=8, steps_per_day=24, transit_amont={})

    for pos in range(32, 40):
        assert result["debit_amont_A_lag8h"].iloc[pos] == pos - 8
        assert result["debit_amont_B_lag8h"].iloc[pos] == pos - 8 + 1000.0
        assert result["debit_amont_A_gradient_24h"].iloc[pos] == 24.0
        assert result["debit_amont_B_gradient_24h"].iloc[pos] == 24.0
