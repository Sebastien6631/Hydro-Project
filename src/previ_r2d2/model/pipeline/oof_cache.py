"""OOF + cache pour l'entraînement -- port fidèle de
_load_or_compute_oof_lgbm/_load_or_compute_oof_lstm/_load_or_compute_lgbm_final
(train_meta.py:333-414, Previ_v2), adapté aux fonctions pures déjà portées
(fit_oof/fit_final LightGBM prennent X/y déjà construits, pas un DataFrame
brut + build_features interne comme Previ_v2). Effets de bord réels et
voulus : lecture/écriture de fichiers cache (.npy/.pkl/.pt)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from previ_r2d2.model.architectures.bilstm.model import BiLSTMHydro
from previ_r2d2.model.architectures.lightgbm.training import fit_final, fit_oof

logger = logging.getLogger(__name__)


def load_or_compute_oof_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    mult_poids: float,
    timestep: str,
    output_dir: Path,
    n_splits: int,
    n_trials: int = 30,
    force: bool = False,
) -> np.ndarray:
    """Charge oof_lgbm.npy si présent et de longueur cohérente avec X, sinon
    calcule fit_oof et sauvegarde. `data_preparation.csv` grandit en continu
    (nouveau débit à chaque rafraîchissement) -- un cache d'une longueur
    différente de X reflète un historique périmé, jamais réutilisable tel
    quel (désalignement débit/features)."""
    path = output_dir / "oof_lgbm.npy"
    if path.exists() and not force:
        cached = np.load(path)
        if len(cached) == len(X):
            logger.info("OOF LGB -- chargé depuis cache")
            return cached
        logger.warning(
            "OOF LGB -- cache périmé (longueur %d != %d attendu), recalcul forcé", len(cached), len(X)
        )
    oof = fit_oof(X, y, horizon, mult_poids, timestep, n_splits=n_splits, n_trials=n_trials)
    np.save(path, oof)
    logger.info("OOF LGB -- calculé et sauvegardé (%s)", path)
    return oof


def load_or_compute_lgbm_final(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    mult_poids: float,
    timestep: str,
    output_dir: Path,
    n_trials: int = 50,
    force: bool = False,
) -> dict:
    """Charge lgbm_final.pkl si présent et entraîné sur le même nombre de
    lignes que X (champ `n_train`, cf. fit_final), sinon calcule fit_final
    et sauvegarde -- même piège que load_or_compute_oof_lgbm : un modèle
    entraîné sur un historique plus court que l'actuel ne doit jamais être
    réutilisé silencieusement."""
    path = output_dir / "lgbm_final.pkl"
    if path.exists() and not force:
        cached = joblib.load(path)
        if cached.get("n_train") == len(X):
            logger.info("LGB final -- chargé depuis cache")
            return cached
        logger.warning(
            "LGB final -- cache périmé (n_train=%s != %d attendu), recalcul forcé",
            cached.get("n_train"), len(X),
        )
    result = fit_final(X, y, horizon, mult_poids, timestep, n_trials=n_trials)
    joblib.dump(result, path)
    logger.info("LGB final -- sauvegardé (%s)", path)
    return result


def load_or_compute_oof_lstm(
    bilstm: BiLSTMHydro,
    X_seq: np.ndarray,
    y: np.ndarray,
    horizon: int,
    output_dir: Path,
    n_splits: int,
    epochs: int,
    dates=None,
    batch_size: int = 256,
    force: bool = False,
) -> np.ndarray:
    """Charge oof_lstm.npy + recharge les poids/scalers dans bilstm si présent
    et de longueur cohérente avec y, sinon fit_oof et sauvegarde -- même
    piège que load_or_compute_oof_lgbm (data_preparation.csv grandit en
    continu, un cache plus court que y est périmé)."""
    path_oof = output_dir / "oof_lstm.npy"
    path_pt = output_dir / "bilstm.pt"
    if path_oof.exists() and not force:
        cached_len = len(np.load(path_oof))
        if cached_len != len(y):
            logger.warning(
                "OOF LSTM -- cache périmé (longueur %d != %d attendu), recalcul forcé", cached_len, len(y)
            )
            path_oof.unlink()
            return load_or_compute_oof_lstm(
                bilstm, X_seq, y, horizon, output_dir, n_splits, epochs, dates, batch_size, force=True
            )
        if path_pt.exists():
            bilstm.load_state_dict(torch.load(path_pt, map_location="cpu"))
            bilstm.eval()
            scaler_files = sorted(output_dir.glob("bilstm_scaler_*.pkl"))
            bilstm.scalers = [joblib.load(p) for p in scaler_files]
            logger.info("OOF LSTM -- chargé depuis cache + poids rechargés (%d scalers)", len(scaler_files))
        else:
            logger.warning("OOF LSTM cache trouvé mais bilstm.pt absent -- réentraînement forcé")
            path_oof.unlink()
            return load_or_compute_oof_lstm(
                bilstm, X_seq, y, horizon, output_dir, n_splits, epochs, dates, batch_size, force=True
            )
        return np.load(path_oof)
    oof = bilstm.fit_oof(X_seq, y, n_splits=n_splits, epochs=epochs, horizon=horizon, batch_size=batch_size, dates=dates)
    np.save(path_oof, oof)
    torch.save(bilstm.state_dict(), path_pt)
    for k, sc in enumerate(bilstm.scalers):
        joblib.dump(sc, output_dir / f"bilstm_scaler_{k}.pkl")
    logger.info("OOF LSTM -- calculé et sauvegardé (%s)", path_oof)
    return oof
