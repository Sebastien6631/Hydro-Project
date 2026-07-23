"""Séries débit amont brutes -- nommage `debit_amont` (un seul) ou
`debit_amont_{code}` (plusieurs), repris tel quel de `lightgbm_model.py`
(Previ_v2) pour rester compatible avec la pièce D à venir."""

from __future__ import annotations

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.preprocessing.automate.debit_csv import read_debit_csv
from previ_r2d2.preprocessing.debit import station_store


def amont_series(rec: dict) -> dict[str, pd.Series]:
    """Séries `debit_amont` (un seul code) ou `debit_amont_{code}` (plusieurs), dédupliquées."""
    codes = list(dict.fromkeys(c for c in (rec.get("stations_vigicrue_amont") or []) if c))
    if not codes:
        return {}
    dossier = rec["dossier"]
    single = len(codes) == 1
    result = {}
    for code in codes:
        path = config.CENTRALES_DIR / dossier / station_store.filename_for(code, "amont")
        column = "debit_amont" if single else f"debit_amont_{code}"
        result[column] = read_debit_csv(path)
    return result
