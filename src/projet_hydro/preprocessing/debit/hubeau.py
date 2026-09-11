"""Client pour l'API Hub'Eau hydrométrie v2 (observations temps réel).

Endpoint `observations_tr` : mesures récentes (~30 derniers jours) haute
fréquence. `resultat_obs` est en L/s pour la grandeur Q (débit), comme eaufrance.
Doc : https://hubeau.eaufrance.fr/page/api-hydrometrie
"""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"


class HubEauError(RuntimeError):
    """Erreur renvoyée par l'API Hub'Eau (code HTTP inattendu ou réponse invalide)."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"[{status}] {message}")


class HubEauClient:
    """Client léger pour l'API Hub'Eau hydrométrie."""

    def __init__(self, base_url: str = BASE_URL, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "previ-record/1.0 (export hydro)",
            }
        )

    def observations_tr(
        self,
        code_entite: str,
        *,
        grandeur: str = "Q",
        date_debut: str | None = None,
        date_fin: str | None = None,
        sort: str = "asc",
        size: int = 20000,
    ) -> list[dict[str, Any]]:
        """Récupère les observations temps réel (toutes les pages).

        `code_entite` : code station/site (ex. "O020002001").
        `date_debut` / `date_fin` : bornes `AAAA-MM-JJ` (optionnelles).
        Renvoie la liste des observations (`date_obs`, `resultat_obs`, …).
        """
        url = f"{self.base_url}/observations_tr"
        params: dict[str, Any] | None = {
            "code_entite": code_entite,
            "grandeur_hydro": grandeur,
            "sort": sort,
            "size": size,
        }
        if date_debut:
            params["date_debut_obs"] = date_debut
        if date_fin:
            params["date_fin_obs"] = date_fin

        results: list[dict[str, Any]] = []
        while url:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            params = None  # l'URL `next` porte déjà ses paramètres
            if resp.status_code not in (200, 206):
                raise HubEauError(resp.status_code, resp.reason or "Erreur HTTP")
            body = resp.json()
            results.extend(body.get("data", []))
            url = body.get("next")
        return results

    def latest_date(self, code_entite: str, grandeur: str = "Q") -> str | None:
        """Date (`date_obs`) de l'observation la plus récente, ou None."""
        url = f"{self.base_url}/observations_tr"
        resp = self._session.get(
            url,
            params={
                "code_entite": code_entite,
                "grandeur_hydro": grandeur,
                "sort": "desc",
                "size": 1,
            },
            timeout=self.timeout,
        )
        if resp.status_code not in (200, 206):
            raise HubEauError(resp.status_code, resp.reason or "Erreur HTTP")
        data = resp.json().get("data", [])
        return data[0].get("date_obs") if data else None

    def stations_referentiel(self, codes: list[str]) -> dict[str, dict[str, float | None]]:
        """Coordonnées/altitude des stations (`referentiel/stations`, un seul appel batch).

        `codes` : codes station (ex. ["O020002001", "O001531001"]). Renvoie
        {code: {"lat": float, "lon": float, "altitude": float | None}} — les
        codes absents de la réponse Hub'Eau sont absents du résultat.
        """
        url = f"{self.base_url}/referentiel/stations"
        resp = self._session.get(
            url,
            params={"code_station": ",".join(codes), "size": len(codes)},
            timeout=self.timeout,
        )
        if resp.status_code not in (200, 206):
            raise HubEauError(resp.status_code, resp.reason or "Erreur HTTP")
        data = resp.json().get("data", [])
        return {
            item["code_station"]: {
                "lat": item["latitude_station"],
                "lon": item["longitude_station"],
                "altitude": item.get("altitude_ref_alti_station"),
            }
            for item in data
            if "code_station" in item
        }
