"""Lecture/écriture/fusion du Data_Preparation (débit + météo + amont brut,
par centrale) -- même logique d'historisation que
`preprocessing/debit/debit_csv.py`, mais sur un DataFrame multi-colonnes
plutôt qu'une Series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_COL = "Date (TU)"


def read_data_preparation_csv(path: Path) -> pd.DataFrame:
    """Relit un `data_preparation.csv` déjà écrit -- DataFrame vide si absent.

    Même correction tz-aware -> tz-naive que `debit_csv.read_debit_csv` (le
    suffixe `Z` écrit par `write_data_preparation_csv` produit un index tz-aware)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.tz_localize(None)
    return df.set_index(DATE_COL)


def merge_data_preparation(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Le nouveau calcul remplace l'ancien aux mêmes horodatages, le reste de
    l'historique est conservé (même logique que `merge_debit_series`)."""
    combined = pd.concat([existing[~existing.index.isin(new.index)], new])
    return combined.sort_index()


def write_data_preparation_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = DATE_COL
    out.to_csv(path, sep=";", encoding="utf-8-sig", date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.3f")
    return len(out)
