"""Fit du meta-learner Stacking -- port fidèle de la séquence de
train_meta.py:693-719 (Previ_v2), qui contourne StackingHydroModel.fit()
(confirmée morte en prod, sous-projet Stacking). Pas de cache : refit
toujours frais (fitter Ridge/petit LGBM sur des OOF déjà calculées est
rapide, contrairement à fit_oof LightGBM/BiLSTM -- cf. pièce C)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from previ_r2d2.model.architectures.stacking import build_meta_features

logger = logging.getLogger(__name__)


def fit_stacking(
    df_full: pd.DataFrame,
    oof_lgbm_multi: np.ndarray,
    oof_lstm_vals: np.ndarray,
    t_last: np.ndarray,
    y_train: np.ndarray,
    horizon: int,
    meteo_feature_cols: list[str],
    output_dir: Path,
    meta_type: str = "ridge",
) -> dict:
    """Construit les features meta, fit le meta-learner (Ridge/LGBM), sauvegarde meta.pkl/meta_scaler.pkl."""
    meta_X, meta_y, valid = build_meta_features(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon, meteo_feature_cols
    )
    q90_train = float(np.percentile(y_train[~np.isnan(y_train)], 90))

    meta_scaler = StandardScaler()
    meta_X_s = meta_scaler.fit_transform(meta_X)

    if meta_type == "lgbm":
        meta = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15, n_jobs=-1, verbose=-1
        ))
    else:
        meta = Ridge(alpha=1.0)

    logger.info(
        "Meta-learner [%s] -- fit sur %d samples, %d features, H=%d pas",
        meta_type, len(meta_X), meta_X.shape[1], horizon,
    )
    meta.fit(meta_X_s, meta_y)

    joblib.dump(meta, output_dir / "meta.pkl")
    joblib.dump(meta_scaler, output_dir / "meta_scaler.pkl")
    logger.info("Meta-learner [%s] entraîné et sauvegardé -> %s", meta_type, output_dir / "meta.pkl")

    return {
        "meta": meta,
        "meta_scaler": meta_scaler,
        "meta_X": meta_X,
        "meta_y": meta_y,
        "valid": valid,
        "q90_train": q90_train,
    }
