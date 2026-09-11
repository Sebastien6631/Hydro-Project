"""Lecture du CSV débit au format commun du projet (`Date (TU);Valeur (en m³/s)`,
port de `hydro_export.py`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_COL = "Date (TU)"
VALUE_COL = "Valeur (en m³/s)"


def read_debit_csv(path: Path) -> pd.Series:
    """Relit un CSV débit déjà écrit -- Series vide si absent.

    Le suffixe `Z` écrit par les scripts d'export ferait de `pd.to_datetime`
    un index tz-aware ; `utc=True` + `tz_localize(None)` le ramène en
    tz-naive pour rester comparable avec les index (naifs) du reste du pipeline.
    """
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.tz_localize(None)
    return pd.Series(df[VALUE_COL].values, index=df[DATE_COL])
