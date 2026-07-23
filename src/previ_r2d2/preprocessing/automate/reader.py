"""Parsing des fichiers capteurs automate (port de load_tokapi_data.py, sans cache).

Format des fichiers : `<racine_variable>/<AAAA>/<MM>/<JJ>/<HH>.txt`, deux formats
de contenu possibles :
  - original : `JJ/MM/AAAA;HH:MM:SS;valeur; ;` (virgule décimale)
  - hydrospot : `HH:MM:SS  valeur` (point décimal, date déduite du chemin)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_txt_file(path: Path) -> pd.Series:
    """Parse un fichier capteur -> Series indexée par timestamp."""
    year, month, day = path.parts[-4], path.parts[-3], path.parts[-2]
    timestamps = []
    values = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if ";" in line:
            date_part, time_part, value_part = line.split(";")[:3]
            day_str, month_str, year_str = date_part.split("/")
            ts = pd.Timestamp(f"{year_str}-{month_str}-{day_str} {time_part}")
            value = float(value_part.strip().replace(",", "."))
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            time_part, value_part = parts[0], parts[1]
            ts = pd.Timestamp(f"{year}-{month}-{day} {time_part}")
            # Certains fichiers "hydrospot" utilisent quand même la virgule
            # décimale (constaté sur des données réelles Melles en full-history)
            # -- même normalisation que la branche format original.
            value = float(value_part.replace(",", "."))
        timestamps.append(ts)
        values.append(value)

    return pd.Series(values, index=pd.DatetimeIndex(timestamps))


def read_variable(racine: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Concatène les fichiers `<racine>/<AAAA>/<MM>/<JJ>/<HH>.txt` dans [start, end].

    Cible directement les dossiers jour de la fenêtre plutôt que de parcourir
    tout l'arbre (`rglob`) et filtrer après coup -- un historique de plusieurs
    années représente des dizaines de milliers de fichiers par capteur, `rglob`
    les liste tous avant de ne garder que les 2-3 jours utiles.
    """
    parts = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="1D"):
        day_dir = racine / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
        if not day_dir.is_dir():
            continue
        for path in sorted(day_dir.glob("*.txt")):
            parts.append(parse_txt_file(path))

    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    combined = pd.concat(parts).sort_index()
    return combined[(combined.index >= start) & (combined.index <= end)]
