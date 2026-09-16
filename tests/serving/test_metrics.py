from __future__ import annotations

import json

from projet_hydro.common import config
from projet_hydro.serving import metrics, predict_service


def test_refresh_model_gauges_sets_kge_per_dossier(monkeypatch):
    monkeypatch.setattr(
        predict_service, "model_infos",
        lambda: [{"dossier": "touzac_g2_G2", "horizon": 8, "kge_stacking": 0.819}],
    )
    metrics.refresh_model_gauges()
    assert metrics.MODEL_KGE.labels(dossier="touzac_g2_G2")._value.get() == 0.819


def test_refresh_model_gauges_clears_stale_dossiers(tmp_path, monkeypatch):
    # render_latest() rafraîchit aussi les gauges de dérive (phase 4.2), qui
    # lisent logs/drift/ sur le disque -- isoler config.ROOT pour ne pas
    # capter de vrais rapports laissés par un lancement manuel de check-drift.py.
    monkeypatch.setattr(config, "ROOT", tmp_path)
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


def test_render_latest_exposes_prometheus_format(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)  # cf. commentaire ci-dessus
    monkeypatch.setattr(
        predict_service, "model_infos",
        lambda: [{"dossier": "touzac_g2_G2", "horizon": 8, "kge_stacking": 0.819}],
    )
    body, content_type = metrics.render_latest()
    assert b"model_kge_stacking" in body
    assert b'dossier="touzac_g2_G2"' in body
    assert "text/plain" in content_type


def test_refresh_drift_gauges_reads_reports_written_by_check_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    drift_dir = tmp_path / "logs" / "drift"
    drift_dir.mkdir(parents=True)
    (drift_dir / "touzac_g2_G2.json").write_text(
        json.dumps({"dossier": "touzac_g2_G2", "drift_share": 0.33, "dataset_drift": False}), encoding="utf-8"
    )

    metrics.refresh_drift_gauges()

    assert metrics.DATA_DRIFT_SHARE.labels(dossier="touzac_g2_G2")._value.get() == 0.33
    assert metrics.DATA_DRIFT_DETECTED.labels(dossier="touzac_g2_G2")._value.get() == 0.0


def test_refresh_drift_gauges_handles_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)  # logs/drift/ n'existe pas encore
    metrics.refresh_drift_gauges()  # ne doit pas lever
