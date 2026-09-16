from __future__ import annotations

import numpy as np
import pandas as pd

from projet_hydro.monitoring.drift import compute_drift, split_reference_current


def _series(mean, std, n, start, freq="1h"):
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_split_reference_current_anchors_on_last_row():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="1h")  # 60 jours
    df = pd.DataFrame({"debit_m3s": range(len(idx))}, index=idx)

    reference, current = split_reference_current(df, current_days=30)

    assert current.index.min() >= df.index.max() - pd.Timedelta(days=30)
    assert reference.index.max() < current.index.min()
    assert len(reference) + len(current) == len(df)


def test_split_reference_current_empty_input():
    reference, current = split_reference_current(pd.DataFrame())
    assert reference.empty
    assert current.empty


def test_compute_drift_no_drift_when_same_distribution():
    reference = pd.DataFrame({"debit_m3s": _series(10, 2, 500, "2024-01-01")})
    current = pd.DataFrame({"debit_m3s": _series(10, 2, 100, "2024-03-01")})

    report = compute_drift(reference, current)

    assert report["n_columns"] == 1
    assert report["dataset_drift"] is False
    assert "debit_m3s" in report["per_column_pvalue"]


def test_compute_drift_detects_shifted_distribution():
    reference = pd.DataFrame({"debit_m3s": _series(10, 2, 500, "2024-01-01")})
    current = pd.DataFrame({"debit_m3s": _series(40, 2, 100, "2024-03-01")})  # décalage fort

    report = compute_drift(reference, current)

    assert report["n_drifted_columns"] == 1
    assert report["dataset_drift"] is True
    assert report["per_column_pvalue"]["debit_m3s"] < 0.05


def test_compute_drift_threshold_is_a_share_not_a_single_column():
    reference = pd.DataFrame({
        "debit_m3s": _series(10, 2, 500, "2024-01-01"),
        "temperature": _series(15, 5, 500, "2024-01-01"),
        "precipitation": _series(1, 0.5, 500, "2024-01-01"),
    })
    current = pd.DataFrame({
        "debit_m3s": _series(40, 2, 100, "2024-03-01"),      # en dérive
        "temperature": _series(15, 5, 100, "2024-03-01"),    # stable
        "precipitation": _series(1, 0.5, 100, "2024-03-01"),  # stable
    })

    report = compute_drift(reference, current, drift_share_threshold=0.4)

    # 1 colonne sur 3 en dérive (33%) < seuil 40% -> pas de dérive au niveau dataset.
    assert report["n_drifted_columns"] == 1
    assert report["dataset_drift"] is False
