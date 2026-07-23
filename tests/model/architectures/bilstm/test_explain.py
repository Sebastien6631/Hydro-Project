from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.architectures.bilstm.explain import plot_attention_heatmaps
from previ_r2d2.model.architectures.bilstm.model import BiLSTMHydro


def make_fixture(n_seq=30, seq_len=20, horizon=3, n_features=3):
    rng = np.random.default_rng(0)
    X_seq_test = rng.normal(0, 1, (n_seq, seq_len, n_features)).astype(np.float32)
    y_test = rng.normal(1, 0.3, (n_seq, horizon)).astype(np.float32)
    lstm_idx_test = np.arange(seq_len, seq_len + n_seq)
    index = pd.date_range("2026-01-01", periods=seq_len + n_seq + 10, freq="1h")
    df_test = pd.DataFrame({"debit_m3s": rng.normal(10, 1, len(index))}, index=index)
    bilstm = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    return bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len


def test_plot_attention_heatmaps_creates_n_peaks_nonempty_pngs(tmp_path):
    bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len = make_fixture()

    paths = plot_attention_heatmaps(bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len, tmp_path)

    assert len(paths) == 5
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_attention_heatmaps_caps_n_peaks_to_available_samples(tmp_path):
    bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len = make_fixture(n_seq=3)

    paths = plot_attention_heatmaps(bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len, tmp_path, n_peaks=5)

    assert len(paths) == 3


def test_plot_attention_heatmaps_selects_peaks_by_last_step_value(tmp_path):
    bilstm, X_seq_test, _, lstm_idx_test, df_test, seq_len = make_fixture(n_seq=10, horizon=2)
    y_test = np.zeros((10, 2), dtype=np.float32)
    y_test[:, -1] = np.arange(10)  # dernier pas croissant -> les 3 plus grands sont les indices 7,8,9

    paths = plot_attention_heatmaps(bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len, tmp_path, n_peaks=3)

    assert len(paths) == 3
