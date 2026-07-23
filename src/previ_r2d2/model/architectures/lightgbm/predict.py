"""Prédiction LightGBM sur un DataFrame complet -- port fidèle de
_predict_lgbm_full (train_meta.py:417-430, Previ_v2). Réaligne les
prédictions sur l'index de `df` car `build_features` peut dropna des
lignes (lags/gradients avec NaN en tête de série)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.architectures.lightgbm.features import build_features, build_future_features


def predict_lgbm_full(
    lgbm_result: dict,
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon: int,
    transit_amont: dict,
    weather_lags: list[int] | None = None,
) -> np.ndarray:
    """Prédit en espace log via build_features + top_features ; NaN si pas de modèle/features."""
    preds = np.full(len(df), np.nan)
    model = lgbm_result["model"]
    top_features = lgbm_result["top_features"]
    if model is None or not top_features:
        return preds
    X, _, date = build_features(df, exutoire, bv_params, steps_per_day, horizon, transit_amont, weather_lags)
    avail = [f for f in top_features if f in X.columns]
    if not avail:
        return preds
    raw = model.predict(X[avail].values)
    idx_in_df = df.index.get_indexer(date)
    valid = idx_in_df >= 0
    preds[idx_in_df[valid]] = raw[valid]
    return preds


def predict_lgbm_future(
    lgbm_result: dict,
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon: int,
    transit_amont: dict,
    weather_lags: list[int] | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Prédit les horizon prochains pas (espace log) via build_future_features ; NaN si pas de modèle/features/données insuffisantes."""
    X, _, date = build_future_features(df, exutoire, bv_params, steps_per_day, horizon, transit_amont, weather_lags)
    future_dates = pd.DatetimeIndex(date[-horizon:])

    model = lgbm_result["model"]
    top_features = lgbm_result["top_features"]
    if model is None or not top_features or len(X) < horizon:
        return np.full(horizon, np.nan), future_dates

    avail = [f for f in top_features if f in X.columns]
    if not avail:
        return np.full(horizon, np.nan), future_dates

    preds = model.predict(X[avail].iloc[-horizon:].values)
    return preds, future_dates
