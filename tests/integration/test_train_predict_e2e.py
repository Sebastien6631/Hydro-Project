"""Tests bout en bout, sur données réelles (pas de mock) -- preuve empirique
que le pipeline allégé (3 centrales, sources bloquées retirées) fonctionne
encore après la simplification. Lents (entraînement réel, réduit) : marqués
`slow`, exclus de la suite par défaut (`pytest -m "not slow"` implicite si
on lance `pytest tests/` sans ce dossier ; lancer explicitement avec
`pytest tests/integration/ -m slow` ou `pytest -m slow`).

Utilise touzac_g2_G2 (pas encore de modèle en production, contrairement à
apas_G1_G4) et le mode source="frozen" pour la prédiction (aucun appel
réseau, ancré sur la dernière ligne connue de data_preparation.csv --
déterministe, reproductible sur n'importe quelle machine)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_script_e2e", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "train.py"
)
train_script = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train_script)

DOSSIER = "touzac_g2_G2"
HORIZON = 8


def _bv_json() -> dict:
    path = config.CENTRALES_DIR / DOSSIER / "bv.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_train_one_produces_a_candidate_model_for_touzac():
    """Entraînement réel (chemin de code réel, données réelles), hyperparamètres
    réduits pour un temps d'exécution raisonnable en test -- pas un mock."""
    summary = train_script.train_one(
        DOSSIER, HORIZON, epochs=3, n_trials_lgbm=2, n_trials_final=2,
    )

    assert DOSSIER in summary
    assert "kge_candidat=" in summary

    model_dir = config.MODELS_DIR / DOSSIER / f"h{HORIZON}"
    assert (model_dir / "version.json").exists()
    assert (model_dir / "meta_config.json").exists()
    assert (model_dir / "lgbm_final.pkl").exists()
    assert (model_dir / "bilstm.pt").exists()


@pytest.mark.slow
def test_run_prediction_frozen_produces_plausible_prevision_json():
    """Nécessite qu'un modèle en production existe déjà pour (DOSSIER, HORIZON)
    -- dépend de test_train_one_produces_a_candidate_model_for_touzac si lancé
    sur une machine sans modèle promu au préalable (sinon FileNotFoundError
    explicite, cf. run_prediction)."""
    bv_json = _bv_json()
    exutoire = bv_json["exutoire"]
    # `now` n'est utilisé par run_prediction, en mode source="frozen", que pour
    # tronquer prevision.json à partir de l'heure de lancement (la fenêtre de
    # données, elle, est toujours ancrée sur la dernière ligne connue de
    # data_preparation.csv, indépendamment de `now`) -- il doit donc tomber
    # juste après la dernière observation affichée et au plus tard sur le
    # premier point prédit pour que les 8 points obtenus soient exactement
    # l'horizon futur (ni observations passées incluses, ni points futurs
    # tronqués). Valeur alignée sur le contenu actuel, gelé, de
    # centrales/touzac_g2_G2/data_preparation.csv (dernier débit observé non
    # NaN : 2026-07-08 11:00 UTC) -- 100% déterministe tant que ce fichier ne
    # change pas (DVC).
    now = __import__("pandas").Timestamp("2026-07-08 20:00:00")

    result = run_prediction(DOSSIER, HORIZON, exutoire, bv_json, now, source="frozen")

    assert result["centrale"] == DOSSIER
    assert result["horizon"] == HORIZON
    assert len(result["q_entrant_m3s"]) == HORIZON
    assert all(q >= 0 for q in result["q_entrant_m3s"])  # débit physiquement plausible

    prevision_path = config.CENTRALES_DIR / DOSSIER / "prevision.json"
    assert prevision_path.exists()
    prevision = json.loads(prevision_path.read_text(encoding="utf-8"))
    assert len(prevision["run"]["points"]) == HORIZON
    for point in prevision["run"]["points"]:
        assert point["debit"]["entrant"] >= 0
