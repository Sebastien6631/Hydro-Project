from __future__ import annotations

import pandas as pd

from previ_r2d2.preprocessing.data_preparation.validation import (
    ValidationReport,
    validate_data_preparation,
)


def _df(debit, start="2024-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(debit), freq=freq)
    return pd.DataFrame({"debit_m3s": debit}, index=idx)


def _long(value=10.0, days=200):
    n = days * 24
    return _df([value] * n)


def test_ok_on_clean_long_series():
    rep = validate_data_preparation(_long())
    assert rep.ok
    assert rep.errors == []


def test_empty_is_error():
    rep = validate_data_preparation(pd.DataFrame(), "apas_G1_G4")
    assert not rep.ok
    assert "vide" in rep.errors[0]


def test_missing_target_column_is_error():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    rep = validate_data_preparation(pd.DataFrame({"autre": range(10)}, index=idx))
    assert not rep.ok
    assert any("debit_m3s" in e for e in rep.errors)


def test_negative_debit_is_error():
    df = _long()
    df.iloc[5, 0] = -1.0
    rep = validate_data_preparation(df)
    assert not rep.ok
    assert any("négative" in e for e in rep.errors)


def test_duplicate_timestamps_is_error():
    df = _long(days=200)
    df = pd.concat([df, df.iloc[[0]]]).sort_index()
    rep = validate_data_preparation(df)
    assert any("double" in e for e in rep.errors)


def test_all_nan_target_is_error():
    rep = validate_data_preparation(_df([float("nan")] * 300))
    assert not rep.ok


def test_short_history_is_warning_not_error():
    rep = validate_data_preparation(_df([10.0] * 240))  # 10 jours
    assert rep.ok
    assert any("historique" in w for w in rep.warnings)


def test_large_gap_is_warning():
    a = _df([10.0] * 100, start="2024-01-01")
    b = _df([10.0] * 100, start="2024-03-01")
    rep = validate_data_preparation(pd.concat([a, b]))
    assert rep.ok
    assert any("trou" in w for w in rep.warnings)


def test_strict_promotes_warnings_to_errors():
    rep = validate_data_preparation(_df([10.0] * 240), strict=True)
    assert not rep.ok
    assert rep.warnings == []


def test_report_is_dataclass():
    rep = ValidationReport("x")
    assert rep.ok and rep.dossier == "x"
