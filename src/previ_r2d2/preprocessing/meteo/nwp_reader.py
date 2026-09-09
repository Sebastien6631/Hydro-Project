"""Lecture des fichiers météo NWP bruts (port du format Previ_v2).

Un jour passé (J-1 et avant) ne garde, après rétention (sous-projet 1,
`retention.py`), que les échéances 000-023 du run 00h de ce jour -- un seul
fichier par heure, aucun arbitrage multi-run à faire ici (hors périmètre,
cf. spec Full_Data piece A -- réservé à predict_future_meta plus tard).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RENAME = {
    "2t (2 metre temperature)": "temperature",
    "tp (precipitation)": "precipitation",
    "deg0l (zero degree level)": "niveau0",
}


def parse_nwp_file(path: Path) -> pd.DataFrame:
    """Un fichier NWP brut -> DataFrame indexé par (latitude, longitude),
    une ligne par point de grille du fichier."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()  # 2021-2024 : header capitalisé (Latitude, 2t/2T)
    df = df.rename(columns=RENAME)
    df["flow_date"] = pd.to_datetime(df["flow_date"])
    return df.set_index(["latitude", "longitude"])


def read_points(
    racine: Path, points: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Série horaire des points météo pertinents pour une centrale.

    `points` : items `{"id": int, "lat": float, "lon": float}` (bv.json
    `stations_meteo_nwp`). Une colonne par point et par variable ; un point
    absent de la grille NWP est silencieusement omis (pas de crash)."""
    parsed = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="1D"):
        day_dir = racine / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
        if not day_dir.is_dir():
            continue
        for path in sorted(day_dir.glob("BARTHE_ENR_EC_OP_recent_*.csv")):
            try:
                parsed.append(parse_nwp_file(path))
            except Exception as exc:
                logger.warning("%s : fichier NWP illisible (%s), ignoré.", path, exc)
                continue
    if not parsed:
        return pd.DataFrame()
    combined = pd.concat(parsed).sort_index()
    combined.index = combined.index.map(lambda key: (round(key[0], 4), round(key[1], 4)))

    series = {}
    for point in points:
        i, key = point["id"], (round(point["lat"], 4), round(point["lon"], 4))
        if key not in combined.index:
            continue
        matched = combined.loc[[key]].set_index("flow_date")
        matched = matched[["temperature", "precipitation", "niveau0"]].resample("1h").mean()
        series[f"latitude_S{i}"] = pd.Series(point["lat"], index=matched.index)
        series[f"longitude_S{i}"] = pd.Series(point["lon"], index=matched.index)
        series[f"temperature_S{i}"] = matched["temperature"]
        series[f"precipitation_S{i}"] = matched["precipitation"]
        series[f"niveau0_S{i}"] = matched["niveau0"]
    if not series:
        return pd.DataFrame()
    result = pd.DataFrame(series).sort_index()
    return result[(result.index >= start) & (result.index <= end)]
