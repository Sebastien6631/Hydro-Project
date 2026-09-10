"""Préparation des séquences BiLSTM (meta) -- port fidèle de
build_sequences/get_seq_cols_from_config (bilstm_hydro.py:54-181, Previ_v2).
Pas de persistance : get_seq_cols reçoit transit_cfg déjà chargé (pas de
lecture disque, contrairement à Previ_v2 qui lit puissance_config.yaml en
interne). Précondition non gardée : build_sequences suppose que
debit_amont_* est déjà décalé par le délai de transit saisonnier avant
l'appel (fait par l'appelant, sous-projet orchestration entraînement, pas
construit ici) -- même contrat que debit_lag_{h}h pour build_features
(LightGBM)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def get_seq_cols(transit_cfg: dict, df: pd.DataFrame, horizon: int) -> tuple[list[str], int]:
    """Sélectionne les colonnes de séquence (débit, amont triés par transit, météo) et seq_len."""
    amont_cols = []
    max_transit = 0
    for col, seasons in transit_cfg.items():
        if col in df.columns and df[col].notna().sum() > 100:
            amont_cols.append(col)
            max_transit = max(max_transit, max(seasons.values()))

    amont_cols_sorted = sorted(amont_cols, key=lambda c: max(transit_cfg[c].values()))

    meteo_pattern = re.compile(r"^(precipitation|temperature)_S\d+$", re.IGNORECASE)
    meteo_cols = sorted([c for c in df.columns if meteo_pattern.match(c)])

    cols = ["debit_m3s"] + amont_cols_sorted + meteo_cols
    seq_len = max(max_transit * 4, horizon * 3, 48)
    return cols, seq_len


def build_sequences(
    df: pd.DataFrame, seq_len: int, horizon: int, seq_cols: list[str], target_col: str = "debit_m3s"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fenêtres glissantes (historique + futur) pour le BiLSTM, avec zeroing anti-leakage de la cible."""
    all_cols = list(dict.fromkeys(seq_cols + [target_col]))
    df_p = df[all_cols].copy()

    for col in seq_cols:
        if "debit" in col.lower():
            df_p[col] = np.log1p(df_p[col].clip(lower=0))
        elif "precipitation" in col.lower():
            df_p[col] = df_p[col].diff(1).clip(lower=0).rolling(6, min_periods=1).sum()

    idx_dt = pd.DatetimeIndex(df_p.index)
    df_p["hour_sin"] = np.sin(2 * np.pi * idx_dt.hour / 24)
    df_p["hour_cos"] = np.cos(2 * np.pi * idx_dt.hour / 24)
    df_p["doy_sin"] = np.sin(2 * np.pi * idx_dt.dayofyear / 365)
    df_p["doy_cos"] = np.cos(2 * np.pi * idx_dt.dayofyear / 365)
    time_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

    all_feat_cols = seq_cols + time_cols
    target_in_seq_idxs = [j for j, c in enumerate(all_feat_cols) if c == target_col]
    total_len = seq_len + horizon

    X, y_list, idx, t_last_list = [], [], [], []
    target_series = df_p[target_col].values
    for i in range(len(df_p) - total_len + 1):
        full_raw = df_p[all_feat_cols].iloc[i : i + total_len].values
        target_vec = target_series[i + seq_len : i + total_len]
        if np.isnan(full_raw).any() or np.isnan(target_vec).any():
            continue

        hist = full_raw[:seq_len].copy()
        fut = full_raw[seq_len:].copy()
        fut[:, target_in_seq_idxs] = 0.0

        is_fut = np.zeros((total_len, 1), dtype=np.float32)
        is_fut[seq_len:] = 1.0

        window = np.concatenate([np.vstack([hist, fut]), is_fut], axis=1)
        X.append(window)
        y_list.append(target_vec)
        idx.append(i + total_len - 1)
        t_last_list.append(i + seq_len - 1)

    if not X:
        raise ValueError(f"Aucune fenêtre valide -- vérifier seq_cols={seq_cols} et df ({len(df)} lignes)")

    return (
        np.array(X, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(idx, dtype=int),
        np.array(t_last_list, dtype=int),
    )


def build_last_window(
    df: pd.DataFrame, seq_len: int, horizon: int, seq_cols: list[str], target_col: str = "debit_m3s"
) -> np.ndarray:
    """Construit LA dernière fenêtre (seq_len historique + horizon futur) pour une prédiction en direct -- pas de fenêtre glissante multiple, contrairement à build_sequences."""
    all_cols = list(dict.fromkeys(seq_cols + [target_col]))
    df_p = df[all_cols].copy()

    for col in seq_cols:
        if "debit" in col.lower():
            df_p[col] = np.log1p(df_p[col].clip(lower=0))
        elif "precipitation" in col.lower():
            df_p[col] = df_p[col].diff(1).clip(lower=0).rolling(6, min_periods=1).sum()

    idx_dt = pd.DatetimeIndex(df_p.index)
    df_p["hour_sin"] = np.sin(2 * np.pi * idx_dt.hour / 24)
    df_p["hour_cos"] = np.cos(2 * np.pi * idx_dt.hour / 24)
    df_p["doy_sin"] = np.sin(2 * np.pi * idx_dt.dayofyear / 365)
    df_p["doy_cos"] = np.cos(2 * np.pi * idx_dt.dayofyear / 365)
    all_feat_cols = seq_cols + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    target_in_seq_idxs = [j for j, c in enumerate(all_feat_cols) if c == target_col]

    total_len = seq_len + horizon
    window = np.nan_to_num(df_p[all_feat_cols].iloc[-total_len:].values, nan=0.0)
    window[seq_len:, target_in_seq_idxs] = 0.0

    is_fut = np.zeros((total_len, 1), dtype=np.float32)
    is_fut[seq_len:] = 1.0
    full = np.concatenate([window, is_fut], axis=1).astype(np.float32)
    return full[np.newaxis]
