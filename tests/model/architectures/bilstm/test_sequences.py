from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.model.architectures.bilstm.sequences import build_last_window, build_sequences, get_seq_cols


def make_fixture(n=150):
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "debit_m3s": np.arange(n, dtype=float),
            "debit_amont_A": np.arange(n, dtype=float),
            "debit_amont_B": np.full(n, np.nan),
            "precipitation_S1": np.cumsum(rng.uniform(0, 0.5, n)),
            "temperature_S1": 280.0,
            "niveau0_S1": 1500.0,
        },
        index=index,
    )


def test_get_seq_cols_filters_sparse_amont_sorts_by_transit_and_detects_meteo():
    df = make_fixture()
    transit_cfg = {
        "debit_amont_A": {"hiver": 5, "printemps": 3, "ete": 2, "automne": 4},
        "debit_amont_B": {"hiver": 10, "printemps": 8, "ete": 6, "automne": 9},
    }

    cols, seq_len = get_seq_cols(transit_cfg, df, horizon=8)

    # debit_amont_B est tout-NaN (<=100 non-NaN) -> exclue.
    assert cols == ["debit_m3s", "debit_amont_A", "niveau0_S1", "precipitation_S1", "temperature_S1"]
    # seq_len = max(max_transit(5)*4, horizon(8)*3, 48) = max(20, 24, 48) = 48.
    assert seq_len == 48


def test_build_sequences_shapes_and_future_zeroing():
    df = make_fixture()
    seq_cols = ["debit_m3s", "debit_amont_A", "precipitation_S1", "temperature_S1"]

    X, y, idx, t_last = build_sequences(df, seq_len=5, horizon=2, seq_cols=seq_cols)

    # n_features=9 (4 seq_cols+4 temporelles+1 is_future) ; N=143 (144 moins 1 fenêtre invalidée par un NaN initial).
    assert X.shape == (143, 7, 9)
    assert y.shape == (143, 2)
    assert idx.shape == (143,)
    assert t_last.shape == (143,)
    # is_future : 0 sur les 5 premiers pas (historique), 1 sur les 2 derniers (futur).
    assert (X[0, :5, -1] == 0.0).all()
    assert (X[0, 5:, -1] == 1.0).all()
    # debit_m3s (colonne 0, la cible) est zérotée dans la fenêtre future.
    assert (X[0, 5:, 0] == 0.0).all()
    # debit_amont_A (colonne 1, pas la cible) n'est PAS zérotée dans le futur.
    assert not (X[0, 5:, 1] == 0.0).all()


def test_build_sequences_raises_when_no_valid_window():
    df = make_fixture(n=5)
    seq_cols = ["debit_m3s", "debit_amont_A", "precipitation_S1", "temperature_S1"]

    with pytest.raises(ValueError, match="Aucune fenêtre valide"):
        build_sequences(df, seq_len=5, horizon=2, seq_cols=seq_cols)


def test_build_last_window_shape_and_target_zeroed_on_future():
    rng = np.random.default_rng(0)
    n, seq_len, horizon = 100, 20, 3
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    debit = 10 + np.cumsum(rng.normal(0, 0.1, n))
    debit[-horizon:] = np.nan
    df = pd.DataFrame({"debit_m3s": debit, "precipitation_S1": rng.uniform(0, 5, n)}, index=index)
    seq_cols = ["debit_m3s", "precipitation_S1"]

    X_last = build_last_window(df, seq_len, horizon, seq_cols)

    assert X_last.shape == (1, seq_len + horizon, len(seq_cols) + 4 + 1)
    assert not np.isnan(X_last).any()
    assert (X_last[0, seq_len:, 0] == 0.0).all()

