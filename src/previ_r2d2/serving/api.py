"""API d'inférence — FastAPI.

Endpoints :
  GET  /health   — état + centrales servies
  GET  /models   — modèles promus (version, KGE)
  POST /predict  — prévision débit h8 pour une centrale

Lancement : uvicorn previ_r2d2.serving.api:app  (voir docker-compose service `api`).
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.eligibility import has_production_model
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

HORIZON = 8  # périmètre projet : h8 uniquement

app = FastAPI(title="previ-R2-D2 — API de prévision de débit", version="0.1.0")


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


@app.get("/models")
def models() -> dict:
    out = []
    for dossier in _served_dossiers():
        data = json.loads((config.MODELS_DIR / dossier / f"h{HORIZON}" / "version.json").read_text(encoding="utf-8"))
        out.append({"dossier": dossier, "horizon": HORIZON, **data})
    return {"models": out}


@app.post("/predict", response_model=PredictResponse)
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
