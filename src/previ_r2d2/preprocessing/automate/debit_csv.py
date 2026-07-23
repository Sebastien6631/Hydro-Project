"""Lecture/écriture/fusion du CSV débit calculé (`debit_automate.csv`).

Même format que les autres CSV débit du projet (`Date (TU);Valeur (en m³/s)`,
port de `hydro_export.py`), avec historisation : un run n'écrase jamais
l'existant, il fusionne la fenêtre nouvellement calculée avec l'historique déjà écrit.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

DATE_COL = "Date (TU)"
VALUE_COL = "Valeur (en m³/s)"


def select_debit_column(qentrant: pd.DataFrame) -> str:
    """Priorité "_ouv" (ouverture) > "_P" (puissance) -- port de hydroforecast.py:328-341."""
    main_cols = [c for c in qentrant.columns if "_ouv" in c]
    if not main_cols:
        main_cols = [c for c in qentrant.columns if "_P" in c]
    if not main_cols:
        raise ValueError(f"Aucune colonne '_ouv' ni '_P' dans Qentrant : {list(qentrant.columns)}")
    return main_cols[0]


def read_debit_csv(path: Path) -> pd.Series:
    """Relit un `debit_automate.csv` déjà écrit -- Series vide si absent.

    Le suffixe `Z` écrit par `write_debit_csv` ferait de `pd.to_datetime` un
    index tz-aware ; `utc=True` + `tz_localize(None)` le ramène en tz-naive
    pour rester comparable avec les index (naifs) du reste du pipeline.
    """
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.tz_localize(None)
    return pd.Series(df[VALUE_COL].values, index=df[DATE_COL])


def merge_debit_series(existing: pd.Series, new: pd.Series) -> pd.Series:
    """Fusionne l'historique déjà écrit avec la fenêtre nouvellement calculée --
    les nouvelles valeurs remplacent les anciennes aux mêmes horodatages
    (recalcul plus complet), le reste de l'historique est conservé."""
    combined = pd.concat([existing[~existing.index.isin(new.index)], new])
    return combined.sort_index()


def write_debit_csv(series: pd.Series, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([DATE_COL, VALUE_COL])
        for ts, value in series.items():
            writer.writerow([ts.strftime("%Y-%m-%dT%H:%M:%SZ"), f"{value:.3f}"])
    return len(series)
