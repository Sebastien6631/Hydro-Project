"""API d'inférence — FastAPI.

Endpoints :
  GET  /health   — état + centrales servies
  GET  /models   — modèles promus (version, KGE)
  POST /predict  — prévision débit h8 pour une centrale

La logique de prévision (`predict_service.py`) est partagée avec le service
BentoML (phase 3.4) -- cette API ne fait que traduire ses erreurs en HTTP.

Lancement : uvicorn projet_hydro.serving.api:app  (voir docker-compose service `api`).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from projet_hydro.serving import predict_service
from projet_hydro.serving.predict_service import HORIZON, PredictionError

app = FastAPI(title="projet_hydro — API de prévision de débit", version="0.1.0")


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


@app.get("/models")
def models() -> dict:
    return {"models": predict_service.model_infos()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = predict_service.predict_dossier(req.dossier)
    except PredictionError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — pas de stack trace côté client
        raise HTTPException(500, f"échec de la prévision : {exc}") from exc
    return PredictResponse(**result)
