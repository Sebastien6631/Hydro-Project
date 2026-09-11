"""Tests boîte noire de l'API (phase 1). `run_prediction` est mocké — le
modèle lui-même est couvert par tests/model/."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from projet_hydro.common import config
from projet_hydro.serving import api


@pytest.fixture
def client(tmp_path, monkeypatch):
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
    monkeypatch.setattr(api, "has_production_model", lambda d, h: (models / d / f"h{h}" / "version.json").exists())
    monkeypatch.setattr(api, "read_data_preparation_csv", lambda p: pd.DataFrame(
        {"debit_m3s": [10.0, 11.0]}, index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"])
    ))
    monkeypatch.setattr(
        api, "run_prediction",
        lambda *a, **k: {"now": "2024-01-01T01:00:00", "q_stacking_m3s": [1.1, 1.2, 1.3],
                         "q_entrant_m3s": [1.0, 1.1, 1.2]},
    )
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["horizon"] == 8
    assert "touzac_g2_G2" in r.json()["dossiers"]


def test_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    m = r.json()["models"]
    assert m[0]["dossier"] == "touzac_g2_G2"
    assert m[0]["kge_stacking"] == 0.8


def test_predict_ok(client):
    r = client.post("/predict", json={"dossier": "touzac_g2_G2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dossier"] == "touzac_g2_G2"
    assert body["horizon"] == 8
    assert body["points"][0]["lead"] == 0


def test_predict_unknown_centrale(client):
    r = client.post("/predict", json={"dossier": "pas_une_centrale"})
    assert r.status_code == 404


def test_predict_no_model(client, monkeypatch):
    monkeypatch.setattr(api, "has_production_model", lambda d, h: False)
    r = client.post("/predict", json={"dossier": "touzac_g2_G2"})
    assert r.status_code == 404
