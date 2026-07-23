"""Orchestration des features LightGBM (meta) -- port fidèle de
LightGBMModel.build_features (Previ_v2 lightgbm_model.py:306-378, flux
entraînement uniquement). Pas de persistance : fonction de calcul pur,
enchaîne les fonctions de feature engineering déjà portées
(model/features/) puis prépare X/y/Date pour l'entraînement."""

from __future__ import annotations

import pandas as pd

from previ_r2d2.model.features.meteo_hydro import compute_meteo_hydro_features
from previ_r2d2.model.features.debit_autoregressif import compute_debit_autoregressif_features
from previ_r2d2.model.features.amont import compute_amont_features


def build_features(
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon: int,
    transit_amont: dict,
    weather_lags: list[int] | None = None,
    cible_col: str = "debit_m3s",
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Prépare X/y/Date pour l'entraînement LightGBM ; lève si cible_col est absente."""
    if cible_col not in df.columns:
        raise ValueError(f"La colonne '{cible_col}' doit exister dans le DataFrame")

    df = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day, weather_lags)
    df = compute_debit_autoregressif_features(df, horizon, steps_per_day, cible_col)
    df = compute_amont_features(df, horizon, steps_per_day, transit_amont)

    if "Q_fct_P" in df.columns:
        df = df.drop(columns=["Q_fct_P"])

    feature_cols = [c for c in df.columns if c != cible_col]
    df_clean = df.dropna(subset=feature_cols)

    date = df_clean.index
    y = df_clean[cible_col].reset_index(drop=True)
    X = df_clean.drop(columns=[cible_col]).reset_index(drop=True)

    return X, y, date


def build_future_features(
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon: int,
    transit_amont: dict,
    weather_lags: list[int] | None = None,
    cible_col: str = "debit_m3s",
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Comme build_features mais conserve les horizon dernières lignes même si leurs features sont NaN (fenêtre de prédiction future)."""
    if cible_col not in df.columns:
        raise ValueError(f"La colonne '{cible_col}' doit exister dans le DataFrame")

    df = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day, weather_lags)
    df = compute_debit_autoregressif_features(df, horizon, steps_per_day, cible_col)
    df = compute_amont_features(df, horizon, steps_per_day, transit_amont)

    if "Q_fct_P" in df.columns:
        df = df.drop(columns=["Q_fct_P"])

    feature_cols = [c for c in df.columns if c != cible_col]
    df_tail = df.tail(horizon)
    df_clean = df.iloc[:-horizon].dropna(subset=feature_cols)
    df_final = pd.concat([df_clean, df_tail])

    date = df_final.index
    y = df_final[cible_col].reset_index(drop=True)
    X = df_final.drop(columns=[cible_col]).reset_index(drop=True)

    return X, y, date
