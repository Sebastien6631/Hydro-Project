from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "onboarding_check", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "onboarding-check.py"
)
onboarding_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(onboarding_check)

from previ_r2d2.common import daily_report


def _write_general(centrales_dir, records):
    ref = centrales_dir / "REFERENCE"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "config-general.json").write_text(json.dumps(records), encoding="utf-8")


def test_run_skips_already_onboarded_dossier(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(daily_report, "STATE_DIR", tmp_path / "logs" / "daily_sync_state")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = tmp_path / "centrales" / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "bv.json").write_text("{}", encoding="utf-8")
    _write_general(tmp_path / "centrales", [{"dossier": "apas_G1_G4"}])

    exit_code = onboarding_check.run()

    assert exit_code == 0
    entries = daily_report.read_today()
    assert "Aucun nouveau raccordement" in entries[0]["body"]


def test_run_reports_missing_fields_for_new_dossier(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(daily_report, "STATE_DIR", tmp_path / "logs" / "daily_sync_state")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    _write_general(tmp_path / "centrales", [{"dossier": "nouveau_G1", "flex_strategy": ""}])

    exit_code = onboarding_check.run()

    assert exit_code == 0
    entries = daily_report.read_today()
    assert entries[0]["has_errors"] is True
    assert "flex_strategy" in entries[0]["body"]
    assert (tmp_path / "logs" / "dvc_markers" / "onboarding.json").exists()


def test_run_continues_after_one_record_raises(tmp_path, monkeypatch):
    """Un enregistrement qui fait planter missing_fields (config corrompue,
    yaml illisible...) ne doit jamais empêcher la validation des autres
    raccordements -- même isolation par enregistrement que train.py::run()."""
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(daily_report, "STATE_DIR", tmp_path / "logs" / "daily_sync_state")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    _write_general(
        tmp_path / "centrales",
        [{"dossier": "casse_G1"}, {"dossier": "nouveau_G1", "flex_strategy": ""}],
    )

    def fake_missing_fields(rec, *args, **kwargs):
        if rec["dossier"] == "casse_G1":
            raise ValueError("config-raccordement.json illisible")
        return ["flex_strategy"]

    monkeypatch.setattr(onboarding_check, "missing_fields", fake_missing_fields)

    exit_code = onboarding_check.run()

    assert exit_code == 0
    entries = daily_report.read_today()
    assert entries[0]["has_errors"] is True
    body = entries[0]["body"]
    assert "casse_G1" in body and "invalide" in body
    assert "nouveau_G1" in body and "flex_strategy" in body
    assert (tmp_path / "logs" / "dvc_markers" / "onboarding.json").exists()
