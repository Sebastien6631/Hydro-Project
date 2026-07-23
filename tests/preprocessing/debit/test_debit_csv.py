from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.preprocessing.debit.debit_csv import read_debit_csv


def test_read_debit_csv_returns_empty_series_when_file_absent(tmp_path):
    result = read_debit_csv(tmp_path / "absent.csv")

    assert result.empty


def test_read_debit_csv_returns_tz_naive_series(tmp_path):
    path = tmp_path / "debit.csv"
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n"
        "2026-07-08T23:00:00Z;1.234\n"
        "2026-07-09T00:00:00Z;2.500\n",
        encoding="utf-8-sig",
    )

    result = read_debit_csv(path)

    assert result.index.tz is None
    assert result.tolist() == pytest.approx([1.234, 2.5])
    assert list(result.index) == [
        pd.Timestamp("2026-07-08 23:00:00"),
        pd.Timestamp("2026-07-09 00:00:00"),
    ]
