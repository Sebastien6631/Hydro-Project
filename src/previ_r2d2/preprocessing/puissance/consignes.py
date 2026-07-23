"""Lecture et interpolation des ordres de consigne EDF (bridage de puissance).

Port de Previ_v2 write_clean_data_v3.py::ConsignesProcessor — étage 1
uniquement (fusion par priorité + interpolation autour des fenêtres de
consigne). L'étage 2 (détection de chaos, ré-échantillonnage horaire) n'est
pas repris.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from previ_r2d2.common import config


def consigne_prefix_for(source_folder: str) -> str:
    """Préfixe (lowercase) des fichiers de consigne pour ce dossier hydrospot_stream.

    Le nom de fichier consigne est `<Mot1>_<Mot2>_<Groupe>_<hh>_<mm>.true|.false` ;
    `source_folder` (déjà résolu par `puissance_store.find_source_folder`, ex.
    "Castillon_Apas_G1") a exactement ce même préfixe `<Mot1>_<Mot2>_<Groupe>`.
    """
    return source_folder.lower()


def read_consigne_events(
    prefix: str, start: pd.Timestamp, end: pd.Timestamp, *, root: Path | None = None,
) -> tuple[list[tuple[pd.Timestamp, float]], list[pd.Timestamp]]:
    """Lit les événements .true/.false pour `prefix` entre `start` et `end`.

    Renvoie (true_events, false_events) triés par date. `true_events` est une
    liste de (date, puissance_mesurée) ; `false_events` une liste de dates.
    Un fichier illisible ou mal formé est ignoré (pas d'exception) : les
    ordres de consigne ne doivent jamais faire échouer l'export puissance.
    """
    root = root if root is not None else config.CONSIGNES_ROOT
    true_events: list[tuple[pd.Timestamp, float]] = []
    false_events: list[pd.Timestamp] = []

    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        day_dir = root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        if not day_dir.is_dir():
            continue
        for path in day_dir.iterdir():
            name = path.name.lower()
            if not (name.endswith(".true") or name.endswith(".false")):
                continue
            if not name.startswith(prefix + "_"):
                continue
            parts = path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                heure, minute = int(parts[-2]), int(parts[-1])
                dt = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                                   hour=heure, minute=minute)
            except ValueError:
                continue
            if name.endswith(".true"):
                try:
                    tokens = path.read_text(encoding="utf-8").strip().split()
                    valeur = float(tokens[2]) if len(tokens) >= 3 else float(tokens[0])
                except (OSError, ValueError, IndexError):
                    continue
                true_events.append((dt, valeur))
            else:
                false_events.append(dt)

    true_events.sort(key=lambda e: e[0])
    false_events.sort()
    return true_events, false_events


def interpolate_consignes(
    df: pd.DataFrame,
    true_events: list[tuple[pd.Timestamp, float]],
    false_events: list[pd.Timestamp],
    *, date_column: str = "Date", power_column: str = "Puissance",
) -> pd.Series:
    """Interpole linéairement `power_column` entre chaque paire consigne TRUE->FALSE.

    Port de Previ_v2 write_clean_data_v3.py::appliquer_consigne. Pour chaque
    ordre .true suivi du premier .false postérieur, la fenêtre est
    [date_true - 2 min, date_false + 2 min] ; les points strictement à
    l'intérieur sont remplacés par une interpolation linéaire entre les
    puissances mesurées juste avant/après la fenêtre (pas entre les valeurs
    de consigne cible). Renvoie une Series indexée par date des valeurs
    interpolées (vide si aucune paire valide).
    """
    result: dict[pd.Timestamp, float] = {}
    if df.empty or not true_events or not false_events:
        return pd.Series(result, dtype=float)

    dates = df[date_column]
    for date_true, _consigne in true_events:
        date_false = next((d for d in false_events if d > date_true), None)
        if date_false is None:
            continue

        date_before = date_true - pd.Timedelta(minutes=2)
        date_after = date_false + pd.Timedelta(minutes=2)

        before_mask = dates <= date_before
        after_mask = dates >= date_after
        if not before_mask.any() or not after_mask.any():
            continue

        start_idx = df[before_mask].index[-1]
        end_idx = df[after_mask].index[0]
        between_mask = (dates > date_before) & (dates < date_after)
        indices = df[between_mask].index.tolist()
        if not indices:
            continue

        start_power = df.loc[start_idx, power_column]
        end_power = df.loc[end_idx, power_column]
        n = len(indices)
        for i, idx in enumerate(indices):
            ratio = i / (n - 1) if n > 1 else 0.5
            result[dates.loc[idx]] = round(start_power + ratio * (end_power - start_power), 1)

    return pd.Series(result, dtype=float)
