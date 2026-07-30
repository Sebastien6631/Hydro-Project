"""Prédiction test set (BiLSTM + Stacking) -- port fidèle du bloc
prédiction test de train_meta.py (Previ_v2), réutilisant build_meta_features
pour éviter toute duplication de StackingHydroModel.predict()."""

from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.architectures.stacking import build_meta_features, slice_lgbm_multistep


def predict_test_set(
    pred_lstm_test: np.ndarray,
    y_test: np.ndarray,
    lstm_idx_test: np.ndarray,
    t_last_test: np.ndarray,
    n_train: int,
    pred_lgbm_all: np.ndarray,
    df_full_ctx: pd.DataFrame,
    meta,
    meta_scaler,
    horizon: int,
    meteo_feature_cols: list[str],
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prédit Stacking sur le test set (BiLSTM et LGBM déjà prédits côté orchestrateur) en réutilisant build_meta_features (pas de duplication de StackingHydroModel.predict()). Retourne aussi les tableaux multi-step et q_now_v pour éviter tout recalcul côté appelant."""
    t_last_test_abs = t_last_test + n_train
    pred_lgbm_multi_test = slice_lgbm_multistep(pred_lgbm_all, t_last_test_abs, horizon)

    meta_X_test, _, valid_meta = build_meta_features(
        df_full_ctx, pred_lgbm_multi_test, pred_lstm_test, t_last_test_abs, y_test, horizon, meteo_feature_cols
    )
    meta_X_test_s = meta_scaler.transform(meta_X_test)
    pred_stacking_multi_valid = np.expm1(meta.predict(meta_X_test_s)).clip(0)
    pred_stacking_multi = np.full((len(t_last_test_abs), horizon), np.nan)
    pred_stacking_multi[valid_meta] = pred_stacking_multi_valid

    pt_last = pred_lstm_test[:, -1]
    pl_last = pred_lgbm_multi_test[:, -1]
    y_t_last = y_test[:, -1] if y_test.ndim == 2 else y_test

    # valid_meta (build_meta_features) exige TOUS les pas d'horizon non-NaN,
    # pas seulement le dernier -- sans ce & valid_meta, une ligne valide au
    # dernier pas mais NaN à un pas intermédiaire laissait un NaN résiduel
    # dans pred_stacking_multi (jamais rempli par build_meta_features), qui
    # cassait ensuite tout calcul de KGE en aval (kge_stacking = null).
    valid = ~np.isnan(pl_last) & ~np.isnan(pt_last) & ~np.isnan(y_t_last) & valid_meta

    y_t_m3s = np.expm1(y_t_last)
    pl_m3s = np.where(np.isnan(pl_last), np.nan, np.expm1(pl_last).clip(0))
    pt_m3s = np.expm1(pt_last)
    pred_stacking_m3s = pred_stacking_multi[:, -1]

    valid_pos = lstm_idx_test[valid]
    df_sub = df_test.iloc[np.clip(valid_pos, 0, len(df_test) - 1)]

    q_now_raw = df_test["debit_m3s"].values[np.clip(t_last_test, 0, len(df_test) - 1)]

    yt_v = y_t_m3s[valid]
    pl_v = pl_m3s[valid]
    pt_v = pt_m3s[valid]
    stk_v = pred_stacking_m3s[valid]
    q_now_v = q_now_raw[valid]

    return yt_v, pl_v, pt_v, stk_v, df_sub, q_now_v, pred_lgbm_multi_test, pred_lstm_test, pred_stacking_multi, valid
