from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.preprocessing.puissance import cleaning


def test_detect_and_nan_chaos_zones_flags_injected_spike_as_nan():
    idx = pd.date_range("2026-01-01", periods=2 * 24 * 60, freq="min")
    values = np.full(len(idx), 100.0)
    values[500:520] = 500.0
    series = pd.Series(values, index=idx, name="power_output")

    result = cleaning.detect_and_nan_chaos_zones(series)

    assert result.iloc[500:520].isna().all()


def test_detect_and_nan_chaos_zones_leaves_smooth_signal_unchanged():
    idx = pd.date_range("2026-01-01", periods=6 * 60, freq="min")
    values = 100 + np.linspace(0, 5, len(idx))
    series = pd.Series(values, index=idx, name="power_output")

    result = cleaning.detect_and_nan_chaos_zones(series)

    assert result.isna().sum() == 0


def test_clean_and_resample_hourly_removes_spike_via_interpolation():
    idx = pd.date_range("2026-01-01", periods=2 * 24 * 60, freq="min")
    values = np.full(len(idx), 100.0)
    values[500:520] = 500.0
    df = pd.DataFrame({"power_output": values}, index=idx)

    result = cleaning.clean_and_resample_hourly(df)

    assert result["power_output"].min() == 100.0
    assert result["power_output"].max() == 100.0


def test_clean_and_resample_hourly_computes_hourly_median_on_smooth_signal():
    idx = pd.date_range("2026-01-01", periods=6 * 60, freq="min")
    values = 100 + np.linspace(0, 5, len(idx))
    df = pd.DataFrame({"power_output": values}, index=idx)

    result = cleaning.clean_and_resample_hourly(df)

    assert list(result.index) == list(pd.date_range("2026-01-01", periods=6, freq="h"))
    assert result["power_output"].tolist() == [100.41, 101.25, 102.08, 102.92, 103.75, 104.59]


def test_detect_and_nan_chaos_zones_handles_two_independent_spikes():
    idx = pd.date_range("2026-01-01", periods=4 * 24 * 60, freq="min")
    values = np.full(len(idx), 100.0)
    values[500:520] = 500.0
    values[3000:3020] = 500.0
    series = pd.Series(values, index=idx, name="power_output")

    result = cleaning.detect_and_nan_chaos_zones(series)

    assert result.iloc[500:520].isna().all()
    assert result.iloc[3000:3020].isna().all()


def test_clean_and_resample_hourly_removes_two_independent_spikes():
    idx = pd.date_range("2026-01-01", periods=4 * 24 * 60, freq="min")
    values = np.full(len(idx), 100.0)
    values[500:520] = 500.0
    values[3000:3020] = 500.0
    df = pd.DataFrame({"power_output": values}, index=idx)

    result = cleaning.clean_and_resample_hourly(df)

    assert result["power_output"].min() == 100.0
    assert result["power_output"].max() == 100.0
