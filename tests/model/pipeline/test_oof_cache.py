from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projet_hydro.model.architectures.bilstm.model import BiLSTMHydro
from projet_hydro.model.pipeline.oof_cache import (
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


def _forward_cpu(model, X):
    """Forward sur CPU, quel que soit le device où le modèle a fini.

    `fit_oof` laisse le modèle sur son device d'entraînement (cuda si GPU) alors
    que le rechargement depuis le cache se fait sur CPU : comparer les deux tels
    quels introduit l'écart d'arrondi cuDNN vs BLAS (~1e-5 absolu, 2e-4 relatif
    mesuré) et rend le test instable une fois sur deux. On force donc le même
    device des deux côtés -- ce que ce test vérifie, ce sont les POIDS rechargés,
    pas le backend qui les exécute."""
    model.cpu().eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


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
    assert set(result1.keys()) == {"model", "top_features", "q_start", "q90", "q99", "n_train", "training_curve"}
    assert result1["n_train"] == len(X)

    result2 = load_or_compute_lgbm_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_trials=0)

    assert result1["q90"] == result2["q90"]


def test_load_or_compute_oof_lgbm_recomputes_when_cache_length_mismatch(tmp_path):
    """Bug de prod porté : data_preparation.csv grandit en continu, le cache
    OOF n'était jamais invalidé -- un cache de longueur différente de X
    (ex. calculé sur un historique plus court) doit déclencher un recalcul,
    pas être silencieusement réutilisé (désalignement débit/features)."""
    X, y = make_lgbm_fixture(n=150)

    load_or_compute_oof_lgbm(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_splits=2, n_trials=0)

    # Simule un cache périmé : écrit à la main un tableau d'une AUTRE longueur
    # (comme si X avait grandi depuis le dernier calcul).
    np.save(tmp_path / "oof_lgbm.npy", np.full(100, np.nan))

    oof = load_or_compute_oof_lgbm(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_splits=2, n_trials=0)

    assert len(oof) == len(X)


def test_load_or_compute_lgbm_final_recomputes_when_n_train_mismatch(tmp_path):
    X, y = make_lgbm_fixture(n=150)

    result1 = load_or_compute_lgbm_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_trials=0)
    assert result1["n_train"] == 150

    # Simule X qui a grandi depuis le dernier calcul (data_preparation.csv rafraîchi).
    X_grown, y_grown = make_lgbm_fixture(n=200)

    result2 = load_or_compute_lgbm_final(X_grown, y_grown, horizon=1, mult_poids=1.0, timestep="1D", output_dir=tmp_path, n_trials=0)

    assert result2["n_train"] == 200


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
    preds_before = _forward_cpu(model, X_seq[:5])

    # nouvelle instance fraîche (poids aléatoires) -> le cache-hit doit recharger les VRAIS poids entraînés.
    model2 = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = load_or_compute_oof_lstm(model2, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)
    preds_after = _forward_cpu(model2, X_seq[:5])

    assert oof.shape == (300, 2)
    assert len(model2.scalers) == 3  # 2 folds + 1 final
    # Comparaison stricte : les deux forwards tournent sur CPU (cf. _forward_cpu),
    # des poids identiques doivent donner des sorties identiques au bit près.
    np.testing.assert_allclose(preds_before, preds_after)


def test_load_or_compute_oof_lstm_recomputes_when_cache_length_mismatch(tmp_path):
    """Même piège que côté LGBM (#5/#6 porté de prod) : un oof_lstm.npy de
    longueur différente de y (X_seq a grandi depuis) doit être recalculé,
    pas rechargé tel quel."""
    X_seq, y = make_bilstm_fixture(n=300)
    horizon, n_features = 2, 3

    model = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    load_or_compute_oof_lstm(model, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)

    # Simule un cache périmé : longueur différente de y.
    np.save(tmp_path / "oof_lstm.npy", np.full((200, horizon), np.nan, dtype=np.float32))

    model2 = BiLSTMHydro(n_features=n_features, horizon=horizon, hidden_size=4, n_layers=1)
    oof = load_or_compute_oof_lstm(model2, X_seq, y, horizon=horizon, output_dir=tmp_path, n_splits=2, epochs=2, batch_size=32)

    assert oof.shape == (300, 2)


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
