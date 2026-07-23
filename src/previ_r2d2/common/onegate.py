"""Client HTTP pour l'API OneGate (namespace `hydrogrid`).

Client minimal et réutilisable par tous les scripts du projet. Il encapsule :
  - l'URL de base et l'authentification JWT Bearer ;
  - la construction des routes `/hydrogrid/...` ;
  - la gestion d'erreurs (corps `{ "error": ... }`) commune à l'API.

Référence des endpoints : skill `call-hydrogrid` / dépôt OneGate
(`Controller/hydrogrid/**`).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from . import config


class OneGateError(RuntimeError):
    """Erreur renvoyée par l'API OneGate (code HTTP != 2xx)."""

    def __init__(self, status: int, message: str, details: Any = None) -> None:
        self.status = status
        self.details = details
        super().__init__(f"[{status}] {message}")


class OneGateClient:
    """Client léger pour les routes `/hydrogrid` de OneGate."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or config.ONEGATE_BASE_URL).rstrip("/")
        self.token = token if token is not None else config.ONEGATE_TOKEN
        self.timeout = timeout or config.ONEGATE_TIMEOUT
        self._session = requests.Session()

    @property
    def is_configured(self) -> bool:
        """Vrai si un jeton d'accès est renseigné."""
        return bool(self.token)

    # -- Bas niveau -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Exécute une requête et renvoie le JSON décodé.

        `path` est relatif au namespace hydrogrid, ex. `memorandum/all`.
        Lève `OneGateError` si l'API renvoie un code d'erreur.
        """
        url = f"{self.base_url}/hydrogrid/{path.lstrip('/')}"
        resp = self._session.request(
            method,
            url,
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )

        try:
            payload = resp.json() if resp.content else None
        except ValueError:
            payload = None

        if not resp.ok:
            message = "Erreur inconnue"
            details = None
            if isinstance(payload, dict):
                message = payload.get("error", message)
                details = payload.get("details")
            raise OneGateError(resp.status_code, message, details)

        return payload

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json: Any = None) -> Any:
        return self.request("POST", path, json=json)

    def patch(self, path: str, json: Any = None) -> Any:
        return self.request("PATCH", path, json=json)

    def put(self, path: str, json: Any = None) -> Any:
        return self.request("PUT", path, json=json)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)

    # -- Namespace memorandum (raccourcis) --------------------------------

    def memorandum_all(self, view: bool = False) -> Any:
        """Tous les aménagements (arbre complet), ou vue enrichie si `view`."""
        return self.get("memorandum/all/view" if view else "memorandum/all")

    def amenagement(self, name: str, view: bool = False) -> Any:
        """Aménagement complet par nom (slug)."""
        slug = quote(name, safe="")
        suffix = "/view" if view else ""
        return self.get(f"memorandum/amenagement/{slug}{suffix}")

    def select(self, keys: list[str]) -> Any:
        """Projection : tous les aménagements réduits aux `keys` demandées."""
        return self.get("memorandum/select", params={"keys": ",".join(keys)})

    def select_layer(self, layer: str, keys: list[str]) -> Any:
        """Liste à plat de toutes les couches d'un niveau, réduites aux `keys`.

        `layer` ∈ amenagement|centrale|raccordement|groupe|transformateur.
        """
        return self.get(
            f"memorandum/select/layer/{layer}",
            params={"keys": ",".join(keys)},
        )

    def set_value(
        self, uuid: str, key: str, value: Any, comment: str | None = None
    ) -> Any:
        """Modifie une valeur d'une couche (PATCH `value/:uuid`)."""
        body: dict[str, Any] = {"key": key, "value": value}
        if comment:
            body["comment"] = comment
        return self.patch(f"memorandum/value/{uuid}", json=body)
