"""API d'inférence — FastAPI.

Endpoints :
  GET  /health   — état + centrales servies (public, monitoring)
  GET  /models   — modèles promus (version, KGE)      [clé API si activée]
  POST /predict  — prévision débit h8 pour une centrale [clé API si activée]

Sécurisation (phase 3) : clé API optionnelle (`config.API_KEY`, en-tête
`X-API-Key`) + logs structurés par requête (`request_id`, latence). Vide par
défaut (tests/CI inchangés) ; timeouts + rate-limit gérés côté nginx
(`infrastructure/nginx/nginx.conf`), pas dans l'app -- déjà le point de
passage unique (phase 2.5).

Lancement : uvicorn projet_hydro.serving.api:app  (voir docker-compose service `api`).
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from projet_hydro.common import config
from projet_hydro.model.pipeline.eligibility import has_production_model
from projet_hydro.model.pipeline.predict_orchestrator import run_prediction
from projet_hydro.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

HORIZON = 8  # périmètre projet : h8 uniquement

logger = logging.getLogger("projet_hydro.api")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Une ligne JSON par requête (request_id, méthode, route, statut, latence)."""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        }))
        response.headers["X-Request-ID"] = request_id
        return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dépendance FastAPI : rejette si `config.API_KEY` est définie et ne
    correspond pas à l'en-tête `X-API-Key`. Vide (défaut) = pas de vérification."""
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "clé API manquante ou invalide (en-tête X-API-Key)")


app = FastAPI(title="projet_hydro — API de prévision de débit", version="0.1.0")
app.add_middleware(RequestLogMiddleware)


class PredictRequest(BaseModel):
    dossier: str


class Point(BaseModel):
    lead: int
    q_stacking_m3s: float
    q_entrant_m3s: float


class PredictResponse(BaseModel):
    dossier: str
    horizon: int
    now: str
    points: list[Point]


def _served_dossiers() -> list[str]:
    return sorted(p.parent.parent.name for p in config.MODELS_DIR.glob(f"*/h{HORIZON}/version.json"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "horizon": HORIZON, "dossiers": _served_dossiers()}


@app.get("/models", dependencies=[Depends(require_api_key)])
def models() -> dict:
    out = []
    for dossier in _served_dossiers():
        data = json.loads((config.MODELS_DIR / dossier / f"h{HORIZON}" / "version.json").read_text(encoding="utf-8"))
        out.append({"dossier": dossier, "horizon": HORIZON, **data})
    return {"models": out}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    dossier = req.dossier

    if not has_production_model(dossier, HORIZON):
        raise HTTPException(404, f"aucun modèle promu pour {dossier} h{HORIZON}")

    bv_path = config.CENTRALES_DIR / dossier / "bv.json"
    if not bv_path.exists():
        raise HTTPException(404, f"centrale inconnue : {dossier}")
    bv_json = json.loads(bv_path.read_text(encoding="utf-8"))

    # source="frozen" : ancrer `now` sur la dernière ligne du CSV (pas l'horloge).
    dp = read_data_preparation_csv(config.CENTRALES_DIR / dossier / "data_preparation.csv")
    if dp.empty:
        raise HTTPException(404, f"pas de data_preparation.csv pour {dossier}")
    now = dp.index.max()

    try:
        result = run_prediction(dossier, HORIZON, bv_json["exutoire"], bv_json, now, source="frozen")
    except Exception as exc:  # noqa: BLE001 — pas de stack trace côté client
        raise HTTPException(500, f"échec de la prévision : {exc}") from exc

    points = [
        Point(lead=i, q_stacking_m3s=round(s, 3), q_entrant_m3s=round(e, 3))
        for i, (s, e) in enumerate(zip(result["q_stacking_m3s"], result["q_entrant_m3s"]))
    ]
    return PredictResponse(dossier=dossier, horizon=HORIZON, now=result["now"], points=points)
