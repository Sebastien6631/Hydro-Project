"""Logique de prévision, isolée de `api.py` -- sépare le métier (lecture
`bv.json`/`data_preparation.csv`, `has_production_model`) du protocole HTTP.

`PredictionError` est une erreur *utilisateur* (dossier/modèle absent) que
`api.py` traduit en HTTP 404. Toute autre exception reste une erreur serveur
(500)."""

from __future__ import annotations

import json

from projet_hydro.common import config
from projet_hydro.model.pipeline.eligibility import has_production_model
from projet_hydro.model.pipeline.predict_orchestrator import run_prediction
from projet_hydro.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

HORIZON = 8  # périmètre projet : h8 uniquement


class PredictionError(Exception):
    """Dossier/modèle/donnée absent -- erreur utilisateur, pas une panne serveur."""


def served_dossiers() -> list[str]:
    return sorted(p.parent.parent.name for p in config.MODELS_DIR.glob(f"*/h{HORIZON}/version.json"))


def model_infos() -> list[dict]:
    out = []
    for dossier in served_dossiers():
        data = json.loads((config.MODELS_DIR / dossier / f"h{HORIZON}" / "version.json").read_text(encoding="utf-8"))
        out.append({"dossier": dossier, "horizon": HORIZON, **data})
    return out


def predict_dossier(dossier: str) -> dict:
    """Prévision h8 pour `dossier` (source="frozen", 100% reproductible).

    Lève `PredictionError` si le dossier/modèle/donnée est absent ; laisse
    remonter toute autre exception (échec réel de `run_prediction`)."""
    if not has_production_model(dossier, HORIZON):
        raise PredictionError(f"aucun modèle promu pour {dossier} h{HORIZON}")

    bv_path = config.CENTRALES_DIR / dossier / "bv.json"
    if not bv_path.exists():
        raise PredictionError(f"centrale inconnue : {dossier}")
    bv_json = json.loads(bv_path.read_text(encoding="utf-8"))

    dp = read_data_preparation_csv(config.CENTRALES_DIR / dossier / "data_preparation.csv")
    if dp.empty:
        raise PredictionError(f"pas de data_preparation.csv pour {dossier}")
    now = dp.index.max()

    result = run_prediction(dossier, HORIZON, bv_json["exutoire"], bv_json, now, source="frozen")

    points = [
        {"lead": i, "q_stacking_m3s": round(s, 3), "q_entrant_m3s": round(e, 3)}
        for i, (s, e) in enumerate(zip(result["q_stacking_m3s"], result["q_entrant_m3s"]))
    ]
    return {"dossier": dossier, "horizon": HORIZON, "now": result["now"], "points": points}
