"""Mise à jour incrémentale des CSV de débits (extension de la queue).

Regarde le dernier enregistrement du CSV d'une station et ajoute les points
plus récents. Choisit la source la plus fraîche :
  - Hub'Eau `observations_tr` (temps réel) si elle est plus récente que
    l'API eaufrance (agrégée en moyenne horaire, heure en cours exclue) ;
  - sinon eaufrance seule.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import hydro_export
from .eaufrance import EauFranceClient
from .hubeau import HubEauClient


def _parse(ts: str) -> datetime:
    """ISO '...Z' -> datetime (aware UTC)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def read_last_timestamp(csv_path: Path) -> str | None:
    """Dernier `Date (TU)` du CSV, ou None si fichier absent/vide."""
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)  # en-tête
        for row in reader:
            if row and row[0]:
                last = row[0]
    return last


def append_rows(csv_path: Path, rows: list[tuple[str, float | None]]) -> int:
    """Ajoute des lignes (ts, valeur m³/s) au CSV existant. Renvoie le nombre ajouté.

    Ouvre en 'utf-8' (SANS BOM) : le BOM n'est écrit qu'à la création du fichier.
    """
    if not rows:
        return 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for ts, val in rows:
            writer.writerow([ts, "" if val is None else f"{val:.3f}"])
    return len(rows)


def _hubeau_hourly_means(
    observations: list[dict[str, Any]], after: str
) -> list[tuple[str, float]]:
    """Moyennes horaires (m³/s) à partir des obs Hub'Eau (L/s), après `after`.

    L'heure contenant l'observation la plus récente est exclue (incomplète).
    """
    after_dt = _parse(after)
    buckets: dict[datetime, list[float]] = defaultdict(list)
    max_dt: datetime | None = None

    for obs in observations:
        ts, val = obs.get("date_obs"), obs.get("resultat_obs")
        if ts is None or val is None:
            continue
        dt = _parse(ts)
        if max_dt is None or dt > max_dt:
            max_dt = dt
        buckets[dt.replace(minute=0, second=0, microsecond=0)].append(val)

    current_hour = (
        max_dt.replace(minute=0, second=0, microsecond=0) if max_dt else None
    )
    rows: list[tuple[str, float]] = []
    for hour in sorted(buckets):
        if hour <= after_dt or hour == current_hour:
            continue
        mean_ls = sum(buckets[hour]) / len(buckets[hour])
        rows.append((hour.strftime("%Y-%m-%dT%H:%M:%SZ"), mean_ls / 1000.0))
    return rows


def update_station_csv(
    csv_path: Path,
    station: str,
    *,
    end_fr: str,
    eaufrance: EauFranceClient | None = None,
    hubeau: HubEauClient | None = None,
) -> dict[str, Any]:
    """Complète le CSV d'une station avec les points plus récents.

    Renvoie un dict : {added, source, eau_latest, hub_latest, note}.
    `source` est None si aucun CSV n'existe encore.
    """
    last = read_last_timestamp(csv_path)
    if last is None:
        return {
            "added": 0,
            "source": None,
            "note": "pas de CSV existant — lance d'abord hydro-collect",
        }

    eaufrance = eaufrance or EauFranceClient()
    hubeau = hubeau or HubEauClient()
    last_dt = _parse(last)
    start_fr = last_dt.strftime("%d/%m/%Y")

    # Source 1 — eaufrance depuis la date du dernier point.
    eau_data = hydro_export.series_data(eaufrance.series(station, start_fr, end_fr))
    eau_latest = eau_data[-1]["t"] if eau_data else None

    # Source 2 — Hub'Eau : date la plus récente disponible.
    hub_latest = hubeau.latest_date(station, grandeur="Q")

    # Comparatif : n'utiliser Hub'Eau que s'il est plus récent qu'eaufrance.
    use_hub = bool(
        hub_latest and (eau_latest is None or _parse(hub_latest) > _parse(eau_latest))
    )

    if use_hub:
        obs = hubeau.observations_tr(
            station, grandeur="Q", date_debut=last_dt.strftime("%Y-%m-%d"), sort="asc"
        )
        rows = _hubeau_hourly_means(obs, after=last)
        source = "hubeau"
    else:
        rows = [
            (p["t"], p["v"] / 1000.0)
            for p in eau_data
            if p.get("t") and p.get("v") is not None and _parse(p["t"]) > last_dt
        ]
        source = "eaufrance"

    added = append_rows(csv_path, rows)
    return {
        "added": added,
        "source": source,
        "eau_latest": eau_latest,
        "hub_latest": hub_latest,
        "last_before": last,
    }
