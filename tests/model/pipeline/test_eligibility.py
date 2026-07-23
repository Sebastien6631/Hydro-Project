from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from previ_r2d2.model.pipeline import eligibility
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv


def _write_history(nas_data_root, dossier, days):
    index = pd.date_range("2024-01-01", periods=days, freq="1D")
    df = pd.DataFrame({"debit_m3s": 1.0}, index=index)
    write_data_preparation_csv(df, nas_data_root / dossier / "data_preparation.csv")


def test_history_span_days_zero_when_no_file(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)

    assert eligibility.history_span_days("inconnu") == 0.0


def test_history_span_days_matches_written_range(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)
    _write_history(tmp_path, "apas_G1_G4", 400)

    assert eligibility.history_span_days("apas_G1_G4") == pytest.approx(399, abs=1)


def test_not_eligible_below_min_history(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    _write_history(tmp_path, "apas_G1_G4", 100)

    assert eligibility.is_eligible_for_training("apas_G1_G4", 8) is False


def test_eligible_for_first_training_at_12_months_no_model_yet(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    _write_history(tmp_path, "apas_G1_G4", 400)

    assert eligibility.is_eligible_for_training("apas_G1_G4", 8) is True


def test_not_eligible_when_model_exists_and_retrained_recently(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(eligibility, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    _write_history(tmp_path, "apas_G1_G4", 400)

    model_dir = tmp_path / "models" / "apas_G1_G4" / "h8"
    model_dir.mkdir(parents=True)
    (model_dir / "version.json").write_text("{}", encoding="utf-8")
    dvc_markers.write("train_apas_G1_G4_h8")

    assert eligibility.is_eligible_for_training("apas_G1_G4", 8) is False


def test_eligible_when_model_exists_but_retrain_overdue(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(eligibility, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    _write_history(tmp_path, "apas_G1_G4", 400)

    model_dir = tmp_path / "models" / "apas_G1_G4" / "h8"
    model_dir.mkdir(parents=True)
    (model_dir / "version.json").write_text("{}", encoding="utf-8")
    marker_path = tmp_path / "logs" / "dvc_markers" / "train_apas_G1_G4_h8.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    old = (datetime.now() - timedelta(days=45)).isoformat()
    marker_path.write_text(json.dumps({"last_run": old}), encoding="utf-8")

    assert eligibility.is_eligible_for_training("apas_G1_G4", 8) is True
