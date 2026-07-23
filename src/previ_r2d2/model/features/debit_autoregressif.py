"""Features débit autorégressives -- port fidèle du bloc débit autorégressif
de `feature()` (Previ_v2 lightgbm_model.py:710-763). Pas de persistance :
fonction de calcul pur, appelée à la volée par l'entraînement/la prédiction
(sous-projets ultérieurs). Calcul linéaire sur une seule série (pas de
boucle par station, contrairement à `meteo_hydro.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
TREND_REL_EPS = 1e-3


def compute_debit_autoregressif_features(
    df: pd.DataFrame, horizon: int, steps_per_day: int, cible_col: str = "debit_m3s"
) -> pd.DataFrame:
    """Ajoute les features débit autorégressives (lags, gradients, déclin, tendance) ; df inchangé sans cible_col."""
    if cible_col not in df.columns:
        return df

    df = df.copy()
    h = horizon
    spd = steps_per_day

    q95_hist = df[cible_col].rolling(spd * 365, min_periods=spd * 30).quantile(0.95)
    df["max_debit_vu"] = df[cible_col].rolling(spd * 30, min_periods=1).max().clip(upper=q95_hist)

    base_lags = [h, int(1.5 * h), 2 * h, 3 * h, 4 * h]
    weekly_lags = [lag for lag in [spd * 7, spd * 14] if lag not in base_lags]
    all_lags = sorted(set(base_lags + weekly_lags))
    for lag in all_lags:
        df[f"debit_lag_{lag}h"] = df[cible_col].shift(lag)

    df["debit_declin_vs_max"] = df[f"debit_lag_{h}h"] / (df["max_debit_vu"] + EPS)

    grad_names = []
    for l1, l2 in zip(all_lags, all_lags[1:]):
        name = f"debit_gradient_{l1}v{l2}"
        df[name] = df[f"debit_lag_{l1}h"] - df[f"debit_lag_{l2}h"]
        grad_names.append(name)

    for g1, g2 in zip(grad_names, grad_names[1:]):
        seg1 = g1.split("_")[2]
        seg2 = g2.split("_")[2]
        df[f"debit_accel_{seg1}v{seg2}"] = df[g1] - df[g2]

    if grad_names:
        df["debit_trend_rel"] = df[grad_names[0]] / (df[f"debit_lag_{h}h"] + TREND_REL_EPS)

    for l1, l2 in [(all_lags[0], all_lags[1]), (all_lags[2], all_lags[3])]:
        df[f"debit_recession_log_{l1}h"] = np.log1p(df[f"debit_lag_{l2}h"]) - np.log1p(df[f"debit_lag_{l1}h"])

    q_recent = df[f"debit_lag_{h}h"]
    df["debit_baseflow_7j"] = q_recent.ewm(span=spd * 7, min_periods=spd).mean().clip(upper=q95_hist)
    df["debit_baseflow_30j"] = q_recent.ewm(span=spd * 30, min_periods=spd).mean().clip(upper=q95_hist)
    df["debit_ratio_rapide_base"] = df["debit_baseflow_7j"] / (df["debit_baseflow_30j"] + EPS)

    q_roll_mean = q_recent.rolling(spd * 30, min_periods=spd * 7).mean()
    q_roll_std = q_recent.rolling(spd * 30, min_periods=spd * 7).std()
    df["debit_anomalie_zscore"] = (q_recent - q_roll_mean) / (q_roll_std + EPS)
    df["debit_ratio_seasonal"] = q_recent / (q_roll_mean + EPS)

    return df
