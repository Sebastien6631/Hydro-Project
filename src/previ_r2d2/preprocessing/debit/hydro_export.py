"""Utilitaires d'export CSV des séries hydro.eaufrance.fr.

Partagé entre les scripts (hydro-series, hydro-collect) : extraction des points
et écriture du CSV à deux colonnes « Date (TU) » / « Valeur (en m³/s) ».
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

DATE_COL = "Date (TU)"
VALUE_COL = "Valeur (en m³/s)"


def to_m3s(v: Any) -> str:
    """Convertit la valeur brute (L/s, `series.unit == "l"`) en m³/s.

    Ex. 202 -> "0.202". Renvoie "" si la valeur est absente (`null`).
    """
    if v is None:
        return ""
    return f"{v / 1000:.3f}"


def series_data(payload: Any) -> list[dict[str, Any]]:
    """Extrait la liste des points (`series.data`) de la réponse API."""
    series = payload.get("series") if isinstance(payload, dict) else None
    return (series or {}).get("data") or []


def write_csv(payload: Any, path: Path) -> int:
    """Écrit le CSV (Date TU ; Valeur m³/s) et renvoie le nombre de lignes.

    utf-8-sig + délimiteur ';' : ouverture propre dans Excel FR.
    """
    data = series_data(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([DATE_COL, VALUE_COL])
        for point in data:
            writer.writerow([point.get("t"), to_m3s(point.get("v"))])
    return len(data)
