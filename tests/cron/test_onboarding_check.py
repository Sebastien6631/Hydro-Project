from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "onboarding_check", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "onboarding-check.py"
)
onboarding_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(onboarding_check)


def _write_general(centrales_dir, records):
    ref = centrales_dir / "REFERENCE"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "config-general.json").write_text(json.dumps(records), encoding="utf-8")


def test_run_skips_already_onboarded_dossier(tmp_path, monkeypatch, caplog):
    from projet_hydro.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    from projet_hydro.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = tmp_path / "centrales" / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "bv.json").write_text("{}", encoding="utf-8")
    _write_general(tmp_path / "centrales", [{"dossier": "apas_G1_G4"}])

    with caplog.at_level("INFO"):
        exit_code = onboarding_check.run()

    assert exit_code == 0
    assert "Aucun nouveau raccordement" in caplog.text


def test_run_reports_missing_fields_for_new_dossier(tmp_path, monkeypatch, caplog):
    from projet_hydro.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    from projet_hydro.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    _write_general(tmp_path / "centrales", [{"dossier": "nouveau_G1", "flex_strategy": ""}])

    with caplog.at_level("INFO"):
        exit_code = onboarding_check.run()

    assert exit_code == 0
    assert "flex_strategy" in caplog.text
    assert (tmp_path / "logs" / "dvc_markers" / "onboarding.json").exists()


def test_run_continues_after_one_record_raises(tmp_path, monkeypatch, caplog):
    """Un enregistrement qui fait planter missing_fields (config corrompue,
    yaml illisible...) ne doit jamais empêcher la validation des autres
    raccordements -- même isolation par enregistrement que train.py::run()."""
    from projet_hydro.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    from projet_hydro.common import dvc_markers

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

    with caplog.at_level("INFO"):
        exit_code = onboarding_check.run()

    assert exit_code == 0
    assert "casse_G1" in caplog.text and "invalide" in caplog.text
    assert "nouveau_G1" in caplog.text and "flex_strategy" in caplog.text
    assert (tmp_path / "logs" / "dvc_markers" / "onboarding.json").exists()
