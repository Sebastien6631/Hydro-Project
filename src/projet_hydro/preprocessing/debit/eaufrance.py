"""Client pour l'API AJAX hydrométrie de hydro.eaufrance.fr.

Récupère les séries de mesures (débits, hauteurs…) d'une station via l'endpoint
`/stationhydro/ajax/<code>/series`. Rien à voir avec OneGate : source de données
publique distincte.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import requests

BASE_URL = "https://hydro.eaufrance.fr"
DATE_FMT = "%d/%m/%Y"

# Valeurs par défaut correspondant à l'appel de référence (débit moyen horaire).
DEFAULT_VARIABLE = "QmnH"
DEFAULT_VARIABLE_TYPE = "simple_and_interpolated_and_hourly_variable"
DEFAULT_STATUS = "most_valid"

# hydro.eaufrance.fr est connu pour ses pannes transitoires (504, timeouts) —
# retry avec backoff plutôt que d'abandonner à la première erreur. Volontairement
# court (2 tentatives, timeout réduit) : ce script tourne (à terme) toutes les
# heures, une station en panne persistante ne doit pas coûter plusieurs minutes
# à chaque run — un échec total prend ~30-50s au lieu de plusieurs minutes.
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_S = (3,)


class EauFranceError(RuntimeError):
    """Erreur renvoyée par l'API eaufrance (code HTTP != 2xx ou réponse invalide)."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"[{status}] {message}")


class EauFranceClient:
    """Client léger pour l'API AJAX de hydro.eaufrance.fr."""

    def __init__(self, base_url: str = BASE_URL, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # En-têtes façon requête AJAX navigateur (l'endpoint est un endpoint ajax).
        self._session.headers.update(
            {
                "User-Agent": "previ-record/1.0 (export hydro eaufrance)",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    def series(
        self,
        code: str,
        start_at: str,
        end_at: str,
        *,
        variable: str = DEFAULT_VARIABLE,
        variable_type: str = DEFAULT_VARIABLE_TYPE,
        status: str = DEFAULT_STATUS,
        step: str = "",
    ) -> Any:
        """Récupère la série d'une station sur une période.

        `code`     : code station (ex. "O024431101").
        `start_at` / `end_at` : dates au format JJ/MM/AAAA.
        Renvoie le JSON décodé (dict avec `series.data`).
        """
        url = f"{self.base_url}/stationhydro/ajax/{code}/series"
        params = {
            "hydro_series[startAt]": start_at,
            "hydro_series[endAt]": end_at,
            "hydro_series[variableType]": variable_type,
            "hydro_series[simpleAndInterpolatedAndHourlyVariable]": variable,
            "hydro_series[step]": step,
            "hydro_series[statusData]": status,
        }
        headers = {"Referer": f"{self.base_url}/stationhydro/{code}"}

        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                if not resp.ok:
                    raise EauFranceError(resp.status_code, resp.reason or "Erreur HTTP")
                try:
                    return resp.json()
                except ValueError as exc:
                    raise EauFranceError(resp.status_code, f"Réponse non-JSON ({exc})")
            except (requests.exceptions.RequestException, EauFranceError):
                if attempt == RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_BACKOFF_S[attempt])


def series_range(client: EauFranceClient, code: str, start_at: str, end_at: str,
                  *, chunk_days: int = 180) -> Any:
    """Récupère une série sur une longue période en la découpant en tranches.

    Un import complet (ex. 01/01/2021 -> aujourd'hui, plusieurs années de données
    horaires) en un seul appel dépasse souvent le délai de réponse de
    hydro.eaufrance.fr (504 Gateway Time-out). Découpe en tranches de
    `chunk_days` jours contiguës et non chevauchantes, concatène les points.
    """
    start = datetime.strptime(start_at, DATE_FMT)
    end = datetime.strptime(end_at, DATE_FMT)

    points: list[Any] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
        payload = client.series(code, chunk_start.strftime(DATE_FMT), chunk_end.strftime(DATE_FMT))
        series = payload.get("series") if isinstance(payload, dict) else None
        points.extend((series or {}).get("data") or [])
        chunk_start = chunk_end + timedelta(days=1)

    return {"series": {"data": points}}
