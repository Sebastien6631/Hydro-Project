"""Fenêtre de prédiction en direct -- assemble dynamiquement débit +
amont + météo (future incluse) via build_dossier, jamais depuis
data_preparation.csv (trop périmé, régénéré à la demande seulement)."""

from __future__ import annotations

import json

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.data_loading import resample_to_daily
from previ_r2d2.preprocessing.data_preparation.dossier_window import build_dossier


def find_record(dossier: str) -> dict:
    """Cherche le raccordement `dossier` dans config-general.json ; lève si absent."""
    general = config.REFERENCE_DIR / "config-general.json"
    records = json.loads(general.read_text(encoding="utf-8"))
    for rec in records:
        if rec["dossier"] == dossier:
            return rec
    raise ValueError(f"Dossier '{dossier}' absent de config-general.json")


def load_prediction_window(
    dossier: str, horizon_steps: int, timestep: str, now: pd.Timestamp, lookback_days: int = 90
) -> pd.DataFrame:
    """Assemble la fenêtre [now - lookback_days, now + horizon_steps] (météo future incluse), resample en 1D si besoin."""
    rec = find_record(dossier)
    step = pd.Timedelta(hours=1) if timestep == "hourly" else pd.Timedelta(days=1)
    start = now - pd.Timedelta(days=lookback_days)
    end = now + horizon_steps * step
    df = build_dossier(rec, start, end)
    if timestep == "1D":
        df = resample_to_daily(df)
    return df
