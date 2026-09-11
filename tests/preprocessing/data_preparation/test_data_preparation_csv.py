from __future__ import annotations

import pandas as pd
import pytest

from projet_hydro.preprocessing.data_preparation.data_preparation_csv import (
    merge_data_preparation,
    read_data_preparation_csv,
    write_data_preparation_csv,
)


def test_read_data_preparation_csv_returns_empty_dataframe_when_file_absent(tmp_path):
    result = read_data_preparation_csv(tmp_path / "absent.csv")

    assert result.empty


def test_write_then_read_data_preparation_csv_roundtrips_as_tz_naive(tmp_path):
    path = tmp_path / "data_preparation.csv"
    df = pd.DataFrame(
        {"debit_m3s": [1.234, 2.5], "temperature_S1": [294.1, 295.0]},
        index=pd.DatetimeIndex(["2026-07-08 23:00:00", "2026-07-09 00:00:00"]),
    )

    write_data_preparation_csv(df, path)
    result = read_data_preparation_csv(path)

    assert result.index.tz is None
    assert result["debit_m3s"].tolist() == pytest.approx([1.234, 2.5])
    assert list(result.index) == [
        pd.Timestamp("2026-07-08 23:00:00"),
        pd.Timestamp("2026-07-09 00:00:00"),
    ]


def test_merge_data_preparation_new_overrides_existing_at_same_timestamp():
    ts = pd.DatetimeIndex(["2026-07-02 00:00:00"])
    existing = pd.DataFrame({"debit_m3s": [2.0]}, index=ts)
    new = pd.DataFrame({"debit_m3s": [99.0]}, index=ts)

    result = merge_data_preparation(existing, new)

    assert result["debit_m3s"].tolist() == pytest.approx([99.0])


def test_merge_data_preparation_preserves_history_outside_new_window():
    existing = pd.DataFrame(
        {"debit_m3s": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-07-01 00:00:00", "2026-07-02 00:00:00"]),
    )
    new = pd.DataFrame({"debit_m3s": [9.0]}, index=pd.DatetimeIndex(["2026-07-03 00:00:00"]))

    result = merge_data_preparation(existing, new)

    assert result["debit_m3s"].tolist() == pytest.approx([1.0, 2.0, 9.0])


def test_merge_data_preparation_handles_new_column_not_in_existing_history():
    existing = pd.DataFrame({"debit_m3s": [1.0]}, index=pd.DatetimeIndex(["2026-07-01 00:00:00"]))
    new = pd.DataFrame(
        {"debit_m3s": [9.0], "temperature_S1": [294.0]},
        index=pd.DatetimeIndex(["2026-07-02 00:00:00"]),
    )

    result = merge_data_preparation(existing, new)

    assert pd.isna(result.loc[pd.Timestamp("2026-07-01"), "temperature_S1"])
    assert result.loc[pd.Timestamp("2026-07-02"), "temperature_S1"] == 294.0


def test_merge_data_preparation_running_twice_does_not_duplicate():
    ts = pd.DatetimeIndex(["2026-07-08 23:00:00"])
    new = pd.DataFrame({"debit_m3s": [2.136]}, index=ts)

    once = merge_data_preparation(pd.DataFrame(), new)
    twice = merge_data_preparation(once, new)

    assert len(twice) == 1
    assert twice["debit_m3s"].tolist() == pytest.approx([2.136])
