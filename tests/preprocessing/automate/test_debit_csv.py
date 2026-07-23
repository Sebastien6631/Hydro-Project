from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.preprocessing.automate.debit_csv import (
    merge_debit_series,
    read_debit_csv,
    select_debit_column,
    write_debit_csv,
)


def test_select_debit_column_prefers_ouv_over_p():
    qentrant = pd.DataFrame({"Q_fct_P": [1.0], "Q_fct_ouv": [2.0]})

    assert select_debit_column(qentrant) == "Q_fct_ouv"


def test_select_debit_column_falls_back_to_p():
    qentrant = pd.DataFrame({"Q_fct_P": [1.0], "Q_fct_charge": [2.0]})

    assert select_debit_column(qentrant) == "Q_fct_P"


def test_select_debit_column_raises_when_neither_present():
    qentrant = pd.DataFrame({"Q_fct_charge": [1.0]})

    with pytest.raises(ValueError):
        select_debit_column(qentrant)


def test_read_debit_csv_returns_empty_series_when_file_absent(tmp_path):
    result = read_debit_csv(tmp_path / "absent.csv")

    assert result.empty


def test_write_then_read_debit_csv_roundtrips_as_tz_naive(tmp_path):
    path = tmp_path / "debit_automate.csv"
    series = pd.Series(
        [1.234, 2.5],
        index=pd.DatetimeIndex(["2026-07-08 23:00:00", "2026-07-09 00:00:00"]),
    )

    write_debit_csv(series, path)
    result = read_debit_csv(path)

    assert result.index.tz is None
    assert result.tolist() == pytest.approx([1.234, 2.5])
    assert list(result.index) == [
        pd.Timestamp("2026-07-08 23:00:00"),
        pd.Timestamp("2026-07-09 00:00:00"),
    ]


def test_merge_debit_series_keeps_existing_history_outside_new_window():
    existing = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2026-07-01 00:00:00", "2026-07-02 00:00:00"]),
    )
    new = pd.Series([9.0], index=pd.DatetimeIndex(["2026-07-03 00:00:00"]))

    result = merge_debit_series(existing, new)

    assert result.tolist() == pytest.approx([1.0, 2.0, 9.0])


def test_merge_debit_series_new_values_override_existing_at_same_timestamp():
    ts = pd.DatetimeIndex(["2026-07-02 00:00:00"])
    existing = pd.Series([2.0], index=ts)
    new = pd.Series([99.0], index=ts)

    result = merge_debit_series(existing, new)

    assert result.tolist() == pytest.approx([99.0])


def test_merge_debit_series_running_twice_does_not_duplicate():
    ts = pd.DatetimeIndex(["2026-07-08 23:00:00"])
    new = pd.Series([2.136], index=ts)

    once = merge_debit_series(pd.Series(dtype=float), new)
    twice = merge_debit_series(once, new)

    assert len(twice) == 1
    assert twice.tolist() == pytest.approx([2.136])
