"""Détection de dérive des données d'entrée (phase 4.2), via Evidently --
compare une fenêtre récente de `data_preparation.csv` à tout l'historique
d'entraînement qui la précède, colonne par colonne (test de
Kolmogorov-Smirnov). Seuil 40 % de colonnes en dérive = dérive du dataset
(cohérent avec l'ancien prototype de ce projet).

Signal de surveillance, pas un contrat bloquant (contrairement à
`preprocessing/data_preparation/validation.py`) : une dérive détectée ne
doit jamais arrêter le pipeline, juste être visible (log + `/metrics`)."""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_SHARE_THRESHOLD = 0.4
CURRENT_WINDOW_DAYS = 30


def split_reference_current(
    df: pd.DataFrame, current_days: int = CURRENT_WINDOW_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`current` = les `current_days` derniers jours ; `reference` = tout ce qui précède."""
    if df.empty:
        return df, df
    cutoff = df.index.max() - pd.Timedelta(days=current_days)
    return df[df.index < cutoff], df[df.index >= cutoff]


def compute_drift(
    reference: pd.DataFrame, current: pd.DataFrame, drift_share_threshold: float = DRIFT_SHARE_THRESHOLD
) -> dict:
    """Rapport de dérive : part de colonnes en dérive + p-value K-S par colonne.

    `reference`/`current` : colonnes numériques uniquement (les colonnes
    catégorielles éventuelles ne sont pas dans `data_preparation.csv`)."""
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    result = report.run(reference_data=reference, current_data=current)
    raw = result.dict()

    n_drifted = 0
    drift_share = 0.0
    per_column: dict[str, float] = {}
    for m in raw["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type") == "evidently:metric_v2:DriftedColumnsCount":
            n_drifted = int(m["value"]["count"])
            drift_share = float(m["value"]["share"])
        elif cfg.get("type") == "evidently:metric_v2:ValueDrift":
            per_column[cfg["column"]] = float(m["value"])

    return {
        "n_columns": len(reference.columns),
        "n_drifted_columns": n_drifted,
        "drift_share": drift_share,
        "dataset_drift": drift_share >= drift_share_threshold,
        "per_column_pvalue": per_column,
    }
