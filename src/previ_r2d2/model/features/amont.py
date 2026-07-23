"""Features amont -- port fidèle du bloc "Stations amont" de `feature()`
(Previ_v2 lightgbm_model.py:770-787), fusionné avec le décalage saisonnier
normalement fait par build_features (lignes 329-347). Pas de persistance :
fonction de calcul pur, appelée à la volée par l'entraînement/la prédiction
(sous-projets ultérieurs)."""

from __future__ import annotations

import pandas as pd

from previ_r2d2.model.features.debit_autoregressif import EPS

SAISONS_MOIS = {
    "hiver": [12, 1, 2],
    "printemps": [3, 4, 5],
    "ete": [6, 7, 8],
    "automne": [9, 10, 11],
}


def shift_amont_columns(df: pd.DataFrame, transit_amont: dict, horizon: int) -> pd.DataFrame:
    """Décale les colonnes debit_amont* par le délai de transit saisonnier (ou horizon simple si absent)."""
    df = df.copy()
    amont_cols = [
        c for c in df.columns
        if c == "debit_amont" or (c.startswith("debit_amont_") and not c.startswith("debit_amont_lag"))
    ]
    for col in amont_cols:
        transit_cfg = transit_amont.get(col, {})
        if transit_cfg:
            original = df[col].copy()
            for saison, mois in SAISONS_MOIS.items():
                transit_h = transit_cfg.get(saison, horizon)
                shift_h = max(horizon, transit_h)
                mask = df.index.month.isin(mois)
                df.loc[mask, col] = original.shift(shift_h)[mask]
        else:
            df[col] = df[col].shift(horizon)
    return df


def compute_amont_features(
    df: pd.DataFrame, horizon: int, steps_per_day: int, transit_amont: dict
) -> pd.DataFrame:
    """Ajoute les features amont (décalage saisonnier, lags, gradients, ratio aval)."""
    df = shift_amont_columns(df, transit_amont, horizon)
    h = horizon
    spd = steps_per_day

    amont_cols = [
        c for c in df.columns
        if c == "debit_amont" or (c.startswith("debit_amont_") and not c.startswith("debit_amont_lag"))
    ]

    for col in amont_cols:
        q_amont = df[col]
        suffix = col
        df[f"{suffix}_lag{h}h"] = q_amont
        for extra in [int(1.5 * h), 2 * h, 3 * h]:
            df[f"{suffix}_lag{h + extra}h"] = q_amont.shift(extra)

        q_lag = df[f"{suffix}_lag{h}h"]
        df[f"{suffix}_gradient_6h"] = q_lag - q_amont.shift(max(1, 6 // (24 // spd)))
        df[f"{suffix}_gradient_24h"] = q_lag - q_amont.shift(max(1, 24 // (24 // spd)))

        df[f"{suffix}_ratio_aval"] = q_lag / (df[f"debit_lag_{h}h"] + EPS)

    return df
