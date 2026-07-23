from __future__ import annotations

import mlflow
import pytest

from previ_r2d2.common import config, mlflow_tracker


def test_get_uri_reads_config_mlflow_uri(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MLFLOW_URI", f"file://{tmp_path}")
    mlflow_tracker.TRACKING_URI = None

    assert mlflow_tracker.get_uri() == f"file://{tmp_path}"


def test_log_training_hybrid_creates_run_with_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_URI", f"file://{tmp_path}")
    mlflow_tracker.TRACKING_URI = None
    mlflow_tracker.MLFLOW_OK = True

    results = {
        "components": {"Stacking": {"kge": 0.8, "r": 0.9, "alpha": 0.95, "beta": 1.0}},
        "kge_by_regime": {}, "kge_by_step": {}, "skill_vs_persistence": {}, "ridge_coef": {},
    }
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    run_id = mlflow_tracker.log_training_hybrid("test_centrale", 72, results, output_dir, params={"epochs": 2})

    assert run_id is not None
    mlflow.set_tracking_uri(mlflow_tracker.get_uri())
    run = mlflow.get_run(run_id)
    assert run.data.metrics["stacking_kge"] == 0.8
    assert run.data.params["epochs"] == "2"


def test_log_prediction_hybrid_creates_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_URI", f"file://{tmp_path}")
    mlflow_tracker.TRACKING_URI = None
    mlflow_tracker.MLFLOW_OK = True

    prediction = {"now": "2026-07-16T12:00:00", "q_entrant_m3s": [1.2, 1.3], "q_stacking_m3s": [1.1, 1.15]}

    run_id = mlflow_tracker.log_prediction_hybrid("test_centrale", 72, prediction)

    assert run_id is not None
    mlflow.set_tracking_uri(mlflow_tracker.get_uri())
    run = mlflow.get_run(run_id)
    assert run.data.metrics["q_entrant_h1_m3s"] == 1.2


def test_log_verification_hybrid_enriches_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_URI", f"file://{tmp_path}")
    mlflow_tracker.TRACKING_URI = None
    mlflow_tracker.MLFLOW_OK = True

    prediction = {"now": "2026-07-16T12:00:00", "q_entrant_m3s": [1.2], "q_stacking_m3s": [1.1]}
    run_id = mlflow_tracker.log_prediction_hybrid("test_centrale", 72, prediction)

    mlflow_tracker.log_verification_hybrid("test_centrale", 72, run_id, {"kge_h1": 0.77})

    mlflow.set_tracking_uri(mlflow_tracker.get_uri())
    run = mlflow.get_run(run_id)
    assert run.data.metrics["reel_kge_h1"] == 0.77


def test_safe_disables_mlflow_after_error_without_raising(monkeypatch):
    mlflow_tracker.MLFLOW_OK = True
    monkeypatch.setattr(mlflow_tracker, "get_uri", lambda: "file:///nonexistent/impossible/path")

    result = mlflow_tracker.log_prediction_hybrid("x", 8, {"now": "n", "q_entrant_m3s": [1.0], "q_stacking_m3s": [1.0]})

    assert result is None
    assert mlflow_tracker.MLFLOW_OK is False
