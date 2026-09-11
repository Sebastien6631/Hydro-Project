"""Série débit brute d'une centrale, lue depuis sa station Hub'Eau de référence
(`config-raccordement.json` / `config-general.json`)."""

from __future__ import annotations

import pandas as pd

from projet_hydro.common import config
from projet_hydro.preprocessing.debit.debit_csv import read_debit_csv
from projet_hydro.preprocessing.debit import station_store


def debit_series(rec: dict) -> pd.Series:
    """Série `debit_m3s` brute pour une centrale, depuis sa station de référence Hub'Eau."""
    dossier = rec["dossier"]
    station = rec.get("station_vigicrue_reference")
    if not station:
        return pd.Series(dtype=float)
    path = config.CENTRALES_DIR / dossier / station_store.filename_for(station, "reference")
    return read_debit_csv(path)
