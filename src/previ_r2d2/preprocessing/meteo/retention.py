"""Nettoyage rétention des fichiers météo NWP grande échéance.

Port de `clean_weather_data.sh` (Previ_v2) : les fichiers
`BARTHE_ENR_EC_OP_recent_{YYYYMMDDHH}_{NNN}.csv` dont l'échéance `NNN`
(en heures) est dans [24, 360] sont supprimés pour les jours passés --
seuls le jour courant et les échéances courtes (<24h) sont conservés.
"""

from __future__ import annotations

import re
from pathlib import Path

FILENAME_RE = re.compile(r"BARTHE_ENR_EC_OP_recent_(\d{10})_(\d{3})\.csv$")
NNN_MIN, NNN_MAX = 24, 360


def files_to_clean(annee_dir: Path, today: str) -> list[Path]:
    """Fichiers de `annee_dir` (récursif) à supprimer : échéance NNN dans
    [24, 360], en excluant `today` (format YYYYMMDD)."""
    result = []
    for path in sorted(annee_dir.rglob("BARTHE_ENR_EC_OP_recent_*.csv")):
        match = FILENAME_RE.search(path.name)
        if not match:
            continue
        date_part, nnn = match.group(1), int(match.group(2))
        if date_part[:8] == today:
            continue
        if NNN_MIN <= nnn <= NNN_MAX:
            result.append(path)
    return result


def clean_year(annee_dir: Path, today: str, *, dry_run: bool = False) -> tuple[list[Path], int]:
    """Supprime (ou simule si `dry_run`) les fichiers de `files_to_clean`.

    Renvoie (fichiers concernés, taille totale en octets).
    """
    targets = files_to_clean(annee_dir, today)
    total_size = sum(p.stat().st_size for p in targets)
    if not dry_run:
        for p in targets:
            p.unlink()
    return targets, total_size
