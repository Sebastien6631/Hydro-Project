from __future__ import annotations

from unittest.mock import patch

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

    # le modèle final chargé (self) reste utilisable en inférence. fit_oof laisse
    # le modèle sur son device d'entraînement (cuda si GPU) : un appel direct à
    # nn.Module.__call__ doit aligner le tenseur, comme n'importe quel code torch.
    model.eval()
    device = next(model.parameters()).device
    x_test = torch.tensor(X_seq[:5], dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(x_test)
    assert preds.shape == (5, horizon)


def test_fit_oof_resumes_from_checkpoint_skipping_completed_folds(tmp_path):
    """Portage d'un fix de prod (#7) : un entraînement long tué en cours de
    route ne doit pas repartir de zéro. Un checkpoint marquant le fold 0
    déjà fait doit faire sauter ce fold au prochain appel -- vérifié en
    comptant les vrais `copy.deepcopy(self)` (un par fold OOF réellement
    entraîné, jamais dans la phase de réentraînement final) : 2 au lieu de
    3 pour n_splits=3."""
    rng = np.random.default_rng(0)
    N, seq_len, horizon, n_features = 300, 10, 2, 3
    X_seq = rng.normal(0, 1, (N, seq_len, n_features)).astype(np.float32)
    y = rng.normal(1, 0.2, (N, horizon)).astype(np.float32)

    checkpoint_path = tmp_path / "oof_lstm.checkpoint.pt"
    torch.save(
        {"oof": np.full((N, horizon), np.nan, dtype=np.float32), "scalers": [], "completed_folds": [0]},
        checkpoint_path,
    )

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)

    import copy as copy_module
    original_deepcopy = copy_module.deepcopy
    calls = []

    def counting_deepcopy(obj, *args, **kwargs):
        if isinstance(obj, BiLSTMHydro):
            calls.append(1)
        return original_deepcopy(obj, *args, **kwargs)

    with patch("previ_r2d2.model.architectures.bilstm.model.copy.deepcopy", side_effect=counting_deepcopy):
        oof = model.fit_oof(
            X_seq, y, n_splits=3, epochs=2, horizon=horizon, batch_size=32,
            checkpoint_path=checkpoint_path,
        )

    assert len(calls) == 2  # fold 0 sauté (déjà dans le checkpoint), folds 1 et 2 réellement entraînés
    assert oof.shape == (N, horizon)
    assert not checkpoint_path.exists()  # nettoyé une fois tous les folds terminés


def test_fit_oof_with_checkpoint_path_cleans_up_after_normal_run(tmp_path):
    rng = np.random.default_rng(0)
    N, seq_len, horizon, n_features = 300, 10, 2, 3
    X_seq = rng.normal(0, 1, (N, seq_len, n_features)).astype(np.float32)
    y = rng.normal(1, 0.2, (N, horizon)).astype(np.float32)
    checkpoint_path = tmp_path / "oof_lstm.checkpoint.pt"

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = model.fit_oof(X_seq, y, n_splits=2, epochs=2, horizon=horizon, batch_size=32, checkpoint_path=checkpoint_path)

    assert oof.shape == (N, horizon)
    assert not checkpoint_path.exists()
