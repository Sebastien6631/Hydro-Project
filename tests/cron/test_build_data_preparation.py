from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_data_preparation_script",
    Path(__file__).resolve().parents[2] / "cron" / "scripts" / "build-data-preparation.py",
)
build_data_preparation_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_data_preparation_script)


def test_run_skips_mail_when_notify_false(tmp_path, monkeypatch):
    """train.py appelle ce script une seule centrale à la fois, juste avant de
    vérifier son éligibilité (cf. bootstrap data_preparation) -- il ne doit
    pas envoyer un mail par centrale (spam), contrairement à son usage cron
    normal (toutes les centrales, un seul mail)."""
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path / "nas")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    reference_dir = tmp_path / "centrales" / "REFERENCE"
    reference_dir.mkdir(parents=True)
    (reference_dir / "config-general.json").write_text(json.dumps([]), encoding="utf-8")

    sent = []
    monkeypatch.setattr(build_data_preparation_script.mailer, "send_report", lambda *a, **k: sent.append(a))

    exit_code = build_data_preparation_script.run(notify=False)

    assert exit_code == 0
    assert sent == []


def test_run_sends_mail_by_default(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path / "nas")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    reference_dir = tmp_path / "centrales" / "REFERENCE"
    reference_dir.mkdir(parents=True)
    (reference_dir / "config-general.json").write_text(json.dumps([]), encoding="utf-8")

    sent = []
    monkeypatch.setattr(build_data_preparation_script.mailer, "send_report", lambda *a, **k: sent.append(a))

    exit_code = build_data_preparation_script.run()

    assert exit_code == 0
    assert len(sent) == 1
