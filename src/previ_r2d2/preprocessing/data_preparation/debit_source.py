"""Sélection de la source débit brute d'une centrale selon `flex_strategy`
(`config-raccordement.json` / `config-general.json`)."""

from __future__ import annotations

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.preprocessing.debit.debit_csv import read_debit_csv
from previ_r2d2.preprocessing.debit import station_store

AUTOMATE_FILENAME = "debit_automate.csv"


def debit_series(rec: dict) -> pd.Series:
    """Série `debit_m3s` brute pour une centrale -- `debit_automate.csv`
    (sous-projet automate) si HAUTE_CHUTE, sinon la station de référence Hub'Eau."""
    dossier = rec["dossier"]
    if rec.get("flex_strategy") == "HAUTE_CHUTE":
        return read_debit_csv(config.CENTRALES_DIR / dossier / AUTOMATE_FILENAME)

    station = rec.get("station_vigicrue_reference")
    if not station:
        return pd.Series(dtype=float)
    path = config.CENTRALES_DIR / dossier / station_store.filename_for(station, "reference")
    return read_debit_csv(path)
