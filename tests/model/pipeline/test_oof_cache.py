from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from previ_r2d2.model.architectures.bilstm.model import BiLSTMHydro
from previ_r2d2.model.pipeline.oof_cache import (
    load_or_compute_oof_lgbm,
    load_or_compute_lgbm_final,
    load_or_compute_oof_lstm,
)


def make_lgbm_fixture(n=150):
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=n, freq="1D")
    X = pd.DataFrame({"feat_a": rng.normal(0, 1, n), "feat_b": rng.normal(0, 1, n)}, index=index)
    y = pd.Series(10 + np.cumsum(rng.normal(0, 0.1, n)), index=index).clip(lower=1)
    return X, y


def test_load_or_compute_oof_lgbm_caches_to_disk_and_reuses_on_second_call(tmp_path):
    X, y = make_lgbm_fixture()

    oof1 = load_or_compute_oof_lgbm(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_splits=2, n_trials=0)

    assert (tmp_path / "oof_lgbm.npy").exists()

    oof2 = load_or_compute_oof_lgbm(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_splits=2, n_trials=0)

    np.testing.assert_array_equal(oof1, oof2)


def test_load_or_compute_lgbm_final_caches_dict_to_disk(tmp_path):
    X, y = make_lgbm_fixture()

    result1 = load_or_compute_lgbm_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_trials=0)

    assert (tmp_path / "lgbm_final.pkl").exists()
    assert set(result1.keys()) == {"model", "top_features", "q_start", "q90", "q99"}

    result2 = load_or_compute_lgbm_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_trials=0)

    assert result1["q90"] == result2["q90"]


def make_bilstm_fixture(n=300, seq_len=10, horizon=2, n_features=3):
    rng = np.random.default_rng(0)
    X_seq = rng.normal(0, 1, (n, seq_len, n_features)).astype(np.float32)
    y = rng.normal(1, 0.2, (n, horizon)).astype(np.float32)
    return X_seq, y


def test_load_or_compute_oof_lstm_cache_hit_reloads_weights_not_random(tmp_path):
    X_seq, y = make_bilstm_fixture()
    horizon, n_features = 2, 3

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    load_or_compute_oof_lstm(model, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)
    preds_before = model(torch.tensor(X_seq[:5], dtype=torch.float32)).detach().numpy()

    # nouvelle instance fraîche (poids aléatoires) -> le cache-hit doit recharger les VRAIS poids entraînés.
    model2 = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = load_or_compute_oof_lstm(model2, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)
    preds_after = model2(torch.tensor(X_seq[:5], dtype=torch.float32)).detach().numpy()

    assert oof.shape == (300, 2)
    assert len(model2.scalers) == 3  # 2 folds + 1 final
    np.testing.assert_allclose(preds_before, preds_after)


def test_load_or_compute_oof_lstm_retrains_when_pt_file_missing(tmp_path):
    X_seq, y = make_bilstm_fixture()
    horizon, n_features = 2, 3

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    load_or_compute_oof_lstm(model, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)
    (tmp_path / "bilstm.pt").unlink()  # cache partiel/corrompu

    model2 = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = load_or_compute_oof_lstm(model2, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)

    assert oof.shape == (300, 2)
    assert (tmp_path / "bilstm.pt").exists()  # réentraîné et re-sauvegardé
