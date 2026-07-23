"""Meta-learner Stacking (Ridge/LGBM) -- port fidèle des briques réellement
utilisées de stacking_hydro.py (Previ_v2). StackingHydroModel.fit()/predict()
(l'API publique de la classe) sont confirmées mortes en production --
train_meta.py/predict_future_meta.py contournent systématiquement la classe
et appellent ces briques directement + Ridge/LGBMRegressor (sklearn) en
direct. Pas de classe wrapper ici : fonctions pures uniquement. Pas de
persistance."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def slice_lgbm_multistep(oof_lgbm: np.ndarray, t_last: np.ndarray, horizon: int) -> np.ndarray:
    """Extrait (N, horizon) prédictions LGB depuis oof_lgbm, ancrées sur t_last (dernier pas observé)."""
    n = len(t_last)
    result = np.full((n, horizon), np.nan, dtype=np.float32)
    for i, tl in enumerate(t_last):
        start = int(tl) + 1
        end = int(tl) + horizon + 1
        if start >= 0 and end <= len(oof_lgbm):
            result[i] = oof_lgbm[start:end]
    return result


def meteo_cols(df: pd.DataFrame) -> list[str]:
    """Colonnes météo {variable}_S{n} (précipitation/température/niveau0), exclut lat/lon."""
    pattern = re.compile(r"^(precipitation|temperature|niveau0)_S\d+$", re.IGNORECASE)
    return [c for c in df.columns if pattern.match(c)]


def kge_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Retourne les 3 composantes KGE (r, alpha, beta) et le KGE global -- diagnostic, pas la loss d'entraînement."""
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / (np.std(y_true) + 1e-6))
    beta = float(np.mean(y_pred) / (np.mean(y_true) + 1e-6))
    kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"kge": kge, "r": r, "alpha": alpha, "beta": beta}


def build_meteo_lag(df: pd.DataFrame, meteo_cols: list[str], horizon: int) -> np.ndarray:
    """Colonnes météo décalées de horizon, NaN remplacés par 0."""
    cols_present = [c for c in meteo_cols if c in df.columns]
    if not cols_present:
        return np.zeros((len(df), 1))
    return df[cols_present].shift(horizon).fillna(0).values


def safe_slope(x: np.ndarray) -> float:
    """Pente d'une régression linéaire sur les valeurs non-NaN de x (0.0 si moins de 2 points valides)."""
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return 0.0
    t_arr = np.where(mask)[0].astype(np.float64)
    return float(np.polyfit(t_arr, x[mask], 1)[0])


def build_meta_features(
    df_full: pd.DataFrame,
    oof_lgbm_multi: np.ndarray,
    oof_lstm_vals: np.ndarray,
    t_last: np.ndarray,
    y_train: np.ndarray,
    horizon: int,
    meteo_feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit meta_X (N_valid, n_feat) et meta_y (N_valid, horizon) pour le meta-learner Ridge/LGBM."""
    lgbm_2d = oof_lgbm_multi if oof_lgbm_multi.ndim == 2 else oof_lgbm_multi[:, None]
    lstm_2d = oof_lstm_vals if oof_lstm_vals.ndim == 2 else oof_lstm_vals[:, None]
    valid = ~np.any(np.isnan(lgbm_2d), axis=1) & ~np.any(np.isnan(lstm_2d), axis=1)
    lgbm_last = lgbm_2d[:, -1]
    lstm_last = lstm_2d[:, -1]

    pos = t_last[valid]
    q90 = np.percentile(y_train[~np.isnan(y_train)], 90)

    q_now_log = np.log1p(df_full["debit_m3s"].values[pos])

    if hasattr(df_full.index, "month"):
        month = df_full.index.month.values[pos]
    else:
        month = pd.to_datetime(df_full["datetime"]).dt.month.values[pos]

    trend_series = df_full["debit_m3s"].rolling(72, min_periods=12).apply(safe_slope, raw=True).fillna(0)
    trend = trend_series.values[pos]

    meteo_arr = build_meteo_lag(df_full, meteo_feature_cols, horizon)

    divergence = np.nan_to_num(lgbm_last[valid] - lstm_last[valid], nan=0.0)

    base_cols = np.column_stack([
        oof_lgbm_multi[valid],
        oof_lstm_vals[valid],
        divergence,
        q_now_log,
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        (q_now_log > q90).astype(int),
        np.nan_to_num(trend, nan=0.0),
    ])

    meta_X = np.hstack([base_cols, np.nan_to_num(meteo_arr[pos], nan=0.0)])
    meta_y = y_train[valid]
    return meta_X, meta_y, valid


def build_meta_features_live(
    pred_lgbm_future: np.ndarray,
    pred_lstm_future: np.ndarray,
    q_now_log: float,
    month: int,
    q90_train: float,
    trend: float,
    meteo_now: np.ndarray,
) -> np.ndarray:
    """Construit meta_X pour UN SEUL point 'maintenant' -- même ordre de colonnes que build_meta_features (lgbm(H), lstm(H), divergence, q_now_log, sin, cos, crue_flag, trend, meteo), à partir de q90_train déjà stocké (pas de y_train disponible à la prédiction)."""
    lgbm_last = pred_lgbm_future[-1]
    lstm_last = pred_lstm_future[-1]
    divergence = float(np.nan_to_num(lgbm_last - lstm_last, nan=0.0))
    crue_flag = float(q_now_log > q90_train)

    base = np.concatenate([
        np.nan_to_num(pred_lgbm_future, nan=0.0),
        pred_lstm_future,
        np.array([
            divergence, q_now_log,
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
            crue_flag, float(np.nan_to_num(trend, nan=0.0)),
        ]),
    ])
    meta_X = np.concatenate([base, np.nan_to_num(meteo_now, nan=0.0)])
    return meta_X.reshape(1, -1)
