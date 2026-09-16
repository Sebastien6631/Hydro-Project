from __future__ import annotations

from projet_hydro.serving import metrics, predict_service


def test_refresh_model_gauges_sets_kge_per_dossier(monkeypatch):
    monkeypatch.setattr(
        predict_service, "model_infos",
        lambda: [{"dossier": "touzac_g2_G2", "horizon": 8, "kge_stacking": 0.819}],
    )
    metrics.refresh_model_gauges()
    assert metrics.MODEL_KGE.labels(dossier="touzac_g2_G2")._value.get() == 0.819


def test_refresh_model_gauges_clears_stale_dossiers(monkeypatch):
    monkeypatch.setattr(
        predict_service, "model_infos",
        lambda: [{"dossier": "touzac_g2_G2", "horizon": 8, "kge_stacking": 0.819}],
    )
    metrics.refresh_model_gauges()

    # Un dossier qui n'a plus de modèle promu ne doit plus apparaître dans /metrics.
    monkeypatch.setattr(predict_service, "model_infos", lambda: [{"dossier": "apas_G1_G4", "horizon": 8, "kge_stacking": 0.856}])
    body, _ = metrics.render_latest()
    assert b'dossier="apas_G1_G4"' in body
    assert b'dossier="touzac_g2_G2"' not in body


def test_render_latest_exposes_prometheus_format(monkeypatch):
    monkeypatch.setattr(
        predict_service, "model_infos",
        lambda: [{"dossier": "touzac_g2_G2", "horizon": 8, "kge_stacking": 0.819}],
    )
    body, content_type = metrics.render_latest()
    assert b"model_kge_stacking" in body
    assert b'dossier="touzac_g2_G2"' in body
    assert "text/plain" in content_type
