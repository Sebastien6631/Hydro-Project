"""Tests bout en bout, sur données réelles (pas de mock) -- preuve empirique
que le pipeline allégé (3 centrales, sources bloquées retirées) fonctionne
encore après la simplification. Lents (entraînement réel, réduit) : marqués
`slow`, exclus de la suite par défaut (`pytest -m "not slow"` implicite si
on lance `pytest tests/` sans ce dossier ; lancer explicitement avec
`pytest tests/integration/ -m slow` ou `pytest -m slow`).

Utilise touzac_g2_G2 (pas encore de modèle en production, contrairement à
apas_G1_G4) et le mode source="frozen" pour la prédiction (aucun appel
réseau, ancré sur la dernière ligne connue de data_preparation.csv --
déterministe, reproductible sur n'importe quelle machine).

**Effet de bord réel et intentionnel (git commit + tag)** --
`test_train_one_produces_a_candidate_model_for_touzac` appelle le vrai
`train_one` -> `promote_model` (`model/pipeline/promotion.py`), pas un mock :
si le candidat entraîné bat le modèle en production (ou qu'il n'y en a pas
encore), la promotion fait un vrai `dvc add` + `git add` + `git commit` +
`git tag` sur CE dépôt (ex. commit `8daa765`, tag `touzac_g2_G2-h8-v1`).
Relancer ce test créera donc une NOUVELLE version promue (v2, v3, ...) avec
son propre commit/tag réels à chaque exécution -- ce n'est pas un défaut
d'hygiène de test à corriger : c'est délibéré et pédagogiquement utile, ça
démontre concrètement comment le système de production versionne ses
modèles. Ce test n'est donc pas idempotent au sens habituel ; le
comportement attendu quand on le relance est justement de voir apparaître
une nouvelle version."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.bv_config import transit_centrale_from_bv_json
from previ_r2d2.model.pipeline.predict_orchestrator import (
    run_prediction,
    season_for_month,
    to_display_timezone,
)
from previ_r2d2.model.pipeline.predict_window import find_record
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

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


def _now_for_frozen_prediction(bv_json: dict) -> pd.Timestamp:
    """Calcule `now` à partir du contenu ACTUEL de data_preparation.csv plutôt
    que de le figer en dur -- si `data_preparation.csv` est régénéré (flux
    documenté par le README : maj-data.py/build-data-preparation.py), cette
    valeur reste valide au lieu de silencieusement décrocher.

    Reprend le même raisonnement que l'ancienne valeur codée en dur : `now`
    n'est utilisé par `run_prediction`, en mode source="frozen", que pour
    tronquer prevision.json à partir de l'heure de lancement (la fenêtre de
    données, elle, est toujours ancrée sur la dernière ligne connue de
    data_preparation.csv, indépendamment de `now`) -- il doit donc tomber
    juste après la dernière observation affichée et au plus tard sur le
    premier point prédit pour que les `HORIZON` points obtenus soient
    exactement l'horizon futur (ni observations passées incluses, ni points
    futurs tronqués).

    On reproduit ici, avec les mêmes fonctions que `run_prediction`
    (`season_for_month`/`to_display_timezone`/`transit_centrale_from_bv_json`),
    le décalage de transit + la conversion d'affichage appliqués à la
    dernière observation, puis on prend le premier point futur (dernière
    observation + décalage + 1 pas horaire) comme borne : `>= now` inclut
    alors exactement les `HORIZON` points futurs et aucune observation
    passée.
    """
    rec = find_record(DOSSIER)
    path = config.CENTRALES_DIR / DOSSIER / "data_preparation.csv"
    df = read_data_preparation_csv(path)
    last_obs = df["debit_m3s"].dropna().index.max()

    decalage_h = 0
    if rec.get("flex_strategy") != "HAUTE_CHUTE":
        transit_centrale = transit_centrale_from_bv_json(bv_json)
        season = season_for_month(last_obs.month)
        decalage_h = round(transit_centrale.get(season, 0))

    first_future_utc = last_obs + pd.Timedelta(hours=decalage_h) + pd.Timedelta(hours=1)
    return to_display_timezone(pd.DatetimeIndex([first_future_utc]), rec.get("flex_strategy"))[0]


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
    # Dérivé du contenu actuel de data_preparation.csv (cf. docstring de
    # _now_for_frozen_prediction) -- reste correct si ce fichier est
    # régénéré, au lieu d'une valeur figée en dur qui décrocherait
    # silencieusement.
    now = _now_for_frozen_prediction(bv_json)

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
