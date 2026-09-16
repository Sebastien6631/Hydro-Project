"""Tests unitaires de la logique de prévision (predict_service.py), isolée
de l'API HTTP. `run_prediction` est mocké -- le modèle lui-même est couvert
par tests/model/."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from projet_hydro.common import config
from projet_hydro.serving import predict_service


@pytest.fixture
def dossier_dir(tmp_path, monkeypatch):
    centrales = tmp_path / "centrales"
    models = tmp_path / "models"
    (centrales / "touzac_g2_G2").mkdir(parents=True)
    (centrales / "touzac_g2_G2" / "bv.json").write_text(
        json.dumps({"exutoire": {"lat": 44.0, "lon": 1.2}}), encoding="utf-8"
    )
    (centrales / "touzac_g2_G2" / "data_preparation.csv").touch()
    vdir = models / "touzac_g2_G2" / "h8"
    vdir.mkdir(parents=True)
    (vdir / "version.json").write_text(json.dumps({"version": 1, "kge_stacking": 0.8}), encoding="utf-8")

    monkeypatch.setattr(config, "CENTRALES_DIR", centrales)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(
        predict_service, "has_production_model", lambda d, h: (models / d / f"h{h}" / "version.json").exists()
    )
    monkeypatch.setattr(predict_service, "read_data_preparation_csv", lambda p: pd.DataFrame(
        {"debit_m3s": [10.0, 11.0]}, index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"])
    ))
    monkeypatch.setattr(
        predict_service, "run_prediction",
        lambda *a, **k: {"now": "2024-01-01T01:00:00", "q_stacking_m3s": [1.1, 1.2, 1.3],
                         "q_entrant_m3s": [1.0, 1.1, 1.2]},
    )
    return tmp_path


def test_served_dossiers_lists_promoted_models(dossier_dir):
    assert predict_service.served_dossiers() == ["touzac_g2_G2"]


def test_model_infos_reads_version_json(dossier_dir):
    infos = predict_service.model_infos()
    assert infos == [{"dossier": "touzac_g2_G2", "horizon": 8, "version": 1, "kge_stacking": 0.8}]


def test_predict_dossier_ok(dossier_dir):
    result = predict_service.predict_dossier("touzac_g2_G2")
    assert result["dossier"] == "touzac_g2_G2"
    assert result["horizon"] == 8
    assert result["points"][0] == {"lead": 0, "q_stacking_m3s": 1.1, "q_entrant_m3s": 1.0}


def test_predict_dossier_raises_on_unknown_centrale(dossier_dir, monkeypatch):
    # has_production_model=True pour passer ce garde-fou et exercer
    # spécifiquement le check bv.json (sinon "pas_une_centrale" échoue déjà
    # au premier garde-fou, cf. test_predict_dossier_raises_when_no_production_model).
    monkeypatch.setattr(predict_service, "has_production_model", lambda d, h: True)
    with pytest.raises(predict_service.PredictionError, match="inconnue"):
        predict_service.predict_dossier("pas_une_centrale")


def test_predict_dossier_raises_when_no_production_model(dossier_dir, monkeypatch):
    monkeypatch.setattr(predict_service, "has_production_model", lambda d, h: False)
    with pytest.raises(predict_service.PredictionError, match="aucun modèle promu"):
        predict_service.predict_dossier("touzac_g2_G2")
