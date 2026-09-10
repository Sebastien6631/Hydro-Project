"""Assemblage débit + amont + météo (API Météo-France) d'une centrale sur une
fenêtre [start, end] arbitraire -- extrait de build-data-preparation.py
pour être réutilisé par le pipeline de prédiction (fenêtre future) sans
dupliquer la logique d'assemblage."""

from __future__ import annotations

import json

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.amont_source import amont_series
from previ_r2d2.preprocessing.data_preparation.debit_source import debit_series
from previ_r2d2.preprocessing.meteo.open_meteo import read_points


def to_hourly(s: pd.Series) -> pd.Series:
    """Resample horaire, sauf sur une Series vide (RangeIndex, pas de DatetimeIndex)."""
    return s if s.empty else s.resample("1h").mean()


def build_dossier(rec: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Assemble débit + amont + météo d'une centrale sur [start, end].

    `start` est relevé au premier point réel du débit (si postérieur) avant de
    lire amont/météo -- le débit est la variable cible, une donnée météo sans
    débit en face n'a aucun intérêt, et ça évite de scanner des années de
    requêtes météo pour rien."""
    dossier = rec["dossier"]
    debit = to_hourly(debit_series(rec))
    if not debit.empty:
        start = max(start, debit.index.min())
    columns = {"debit_m3s": debit}

    columns.update({name: to_hourly(s) for name, s in amont_series(rec).items()})

    bv_path = config.CENTRALES_DIR / dossier / "bv.json"
    if bv_path.exists():
        bv = json.loads(bv_path.read_text(encoding="utf-8"))
        points = bv.get("stations_meteo_nwp", [])
        if points:
            meteo = read_points(points, start, end)
            overlap = columns.keys() & meteo.columns
            if overlap:
                raise ValueError(f"{dossier} : colonnes en conflit : {overlap}")
            for col in meteo.columns:
                columns[col] = meteo[col]

    df = pd.DataFrame(columns)
    return df[(df.index >= start) & (df.index <= end)]
