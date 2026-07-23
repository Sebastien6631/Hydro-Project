from __future__ import annotations

import numpy as np
import torch

from previ_r2d2.model.architectures.bilstm.model import BiLSTMHydro


def test_forward_multi_horizon_returns_batch_by_horizon_shape():
    model = BiLSTMHydro(n_features=3, horizon=4, hidden_size=8, n_layers=1)
    x = torch.randn(5, 10, 3)

    out = model(x)

    assert out.shape == (5, 4)


def test_forward_horizon_one_squeezes_to_scalar_per_batch():
    model = BiLSTMHydro(n_features=3, horizon=1, hidden_size=8, n_layers=1)
    x = torch.randn(5, 10, 3)

    out = model(x)

    assert out.shape == (5,)


def test_get_attention_weights_shape_and_sums_to_one():
    model = BiLSTMHydro(n_features=3, horizon=4, hidden_size=8, n_layers=1)
    x = torch.randn(5, 10, 3)

    attn = model.get_attention_weights(x)

    assert attn.shape == (5, 10)
    assert np.allclose(attn.sum(axis=1), 1.0)


def test_predict_normalizes_with_last_scaler_and_returns_numpy_array():
    model = BiLSTMHydro(n_features=3, horizon=2, hidden_size=4, n_layers=1)
    rng = np.random.default_rng(0)
    X_seq = rng.normal(0, 1, (5, 10, 3)).astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_seq.reshape(-1, 3))
    model.scalers = [scaler]

    preds = model.predict(X_seq)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5, 2)


def test_predict_without_scalers_uses_raw_input():
    model = BiLSTMHydro(n_features=3, horizon=2, hidden_size=4, n_layers=1)
    rng = np.random.default_rng(0)
    X_seq = rng.normal(0, 1, (5, 10, 3)).astype(np.float32)

    preds = model.predict(X_seq)

    assert preds.shape == (5, 2)


def test_fit_oof_returns_correct_shape_and_trains_usable_model():
    rng = np.random.default_rng(0)
    N, seq_len, horizon, n_features = 300, 10, 2, 3
    X_seq = rng.normal(0, 1, (N, seq_len, n_features)).astype(np.float32)
    y = rng.normal(1, 0.2, (N, horizon)).astype(np.float32)

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = model.fit_oof(X_seq, y, n_splits=2, epochs=2, horizon=horizon, batch_size=32)

    assert oof.shape == (N, horizon)
    assert np.sum(~np.isnan(oof)) > 0
    # un scaler par fold OOF (2) + un scaler final = 3.
    assert len(model.scalers) == 3

    # le modèle final chargé (self) reste utilisable en inférence.
    model.eval()
    x_test = torch.tensor(X_seq[:5], dtype=torch.float32)
    with torch.no_grad():
        preds = model(x_test)
    assert preds.shape == (5, horizon)
