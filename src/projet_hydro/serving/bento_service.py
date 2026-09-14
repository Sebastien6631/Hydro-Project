"""Service BentoML (phase 3.4) -- démontre le serving via BentoML, en plus de
l'API FastAPI (phase 1). Réutilise `predict_service.py` (même logique, pas de
duplication) : deux surfaces de serving, un seul code métier.

Lancement : bentoml serve projet_hydro.serving.bento_service:PredictionService
(voir docker-compose service `bento`, Dockerfile.bento -- image séparée de
l'API/des tests, bentoml n'est pas une dépendance du cœur du projet)."""

from __future__ import annotations

import bentoml

from projet_hydro.serving import predict_service
from projet_hydro.serving.predict_service import PredictionError


@bentoml.service(name="projet_hydro_predict", resources={"cpu": "1"})
class PredictionService:
    @bentoml.api
    def health(self) -> dict:
        return {
            "status": "ok",
            "horizon": predict_service.HORIZON,
            "dossiers": predict_service.served_dossiers(),
        }

    @bentoml.api
    def models(self) -> dict:
        return {"models": predict_service.model_infos()}

    @bentoml.api
    def predict(self, dossier: str) -> dict:
        try:
            return predict_service.predict_dossier(dossier)
        except PredictionError as exc:
            raise bentoml.exceptions.NotFound(str(exc)) from exc
