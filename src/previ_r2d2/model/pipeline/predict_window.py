"""Fenêtre de prédiction -- assemble débit + amont + météo (future incluse)
via build_dossier ("live", défaut), ou relit directement data_preparation.csv
("frozen", ancré sur sa dernière ligne connue) -- 100% reproductible, aucun
appel réseau, utilisé par les tests bout en bout."""

from __future__ import annotations

import json

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.data_loading import resample_to_daily
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv
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
    dossier: str, horizon_steps: int, timestep: str, now: pd.Timestamp,
    lookback_days: int = 90, source: str = "live",
) -> pd.DataFrame:
    """Assemble la fenêtre [now - lookback_days, now + horizon_steps], resample en 1D si besoin.

    `source="live"` (défaut) : assemble dynamiquement via build_dossier (débit
    à jour Hub'Eau, météo dégradée en attendant l'API météo publique).
    `source="frozen"` : relit data_preparation.csv tel quel, `now` est alors
    ignoré et remplacé par la dernière ligne connue du fichier -- 100%
    reproductible, aucun appel réseau (utilisé par les tests bout en bout)."""
    step = pd.Timedelta(hours=1) if timestep == "hourly" else pd.Timedelta(days=1)

    if source == "frozen":
        path = config.CENTRALES_DIR / dossier / "data_preparation.csv"
        df = read_data_preparation_csv(path)
        if df.empty:
            return df
        anchor = df.index.max()
        start = anchor - pd.Timedelta(days=lookback_days)
        end = anchor + horizon_steps * step
        df = df[(df.index >= start) & (df.index <= end)]
    else:
        rec = find_record(dossier)
        start = now - pd.Timedelta(days=lookback_days)
        end = now + horizon_steps * step
        df = build_dossier(rec, start, end)

    if timestep == "1D":
        df = resample_to_daily(df)
    return df
