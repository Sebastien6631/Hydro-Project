"""API d'inférence — FastAPI.

Endpoints :
  GET  /health   — état + centrales servies (public, monitoring)
  GET  /models   — modèles promus (version, KGE)      [clé API si activée]
  POST /predict  — prévision débit h8 pour une centrale [clé API si activée]

Sécurisation (phase 3.3) : clé API optionnelle (`config.API_KEY`, en-tête
`X-API-Key`) + logs structurés par requête (`request_id`, latence). Vide par
défaut (tests/CI inchangés) ; timeouts + rate-limit gérés côté nginx
(`infrastructure/nginx/nginx.conf`), pas dans l'app -- déjà le point de
passage unique (phase 2.5).

La logique de prévision (`predict_service.py`, phase 3.4) est partagée avec
le service BentoML -- cette API ne fait que traduire ses erreurs en HTTP.

Monitoring (phase 4.1) : `GET /metrics` (Prometheus, `serving/metrics.py`) --
public comme `/health`, scrapé en interne par Prometheus, jamais par un
client de l'API.

Lancement : uvicorn projet_hydro.serving.api:app  (voir docker-compose service `api`).
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from projet_hydro.common import config
from projet_hydro.serving import metrics, predict_service
from projet_hydro.serving.predict_service import HORIZON, PredictionError

logger = logging.getLogger("projet_hydro.api")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Une ligne JSON par requête (request_id, méthode, route, statut, latence)."""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        duration_ms = round(duration * 1000, 1)
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        }))
        response.headers["X-Request-ID"] = request_id
        metrics.REQUESTS_TOTAL.labels(
            method=request.method, path=request.url.path, status=response.status_code
        ).inc()
        metrics.REQUEST_DURATION.labels(method=request.method, path=request.url.path).observe(duration)
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "horizon": HORIZON, "dossiers": predict_service.served_dossiers()}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    body, content_type = metrics.render_latest()
    return Response(content=body, media_type=content_type)


@app.get("/models", dependencies=[Depends(require_api_key)])
def models() -> dict:
    return {"models": predict_service.model_infos()}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = predict_service.predict_dossier(req.dossier)
    except PredictionError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — pas de stack trace côté client
        raise HTTPException(500, f"échec de la prévision : {exc}") from exc
    return PredictResponse(**result)
