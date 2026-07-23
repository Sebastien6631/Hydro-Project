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


def test_run_logs_summary(tmp_path, monkeypatch, caplog):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "REFERENCE_DIR", tmp_path / "centrales" / "REFERENCE")
    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path / "nas")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    reference_dir = tmp_path / "centrales" / "REFERENCE"
    reference_dir.mkdir(parents=True)
    (reference_dir / "config-general.json").write_text(json.dumps([]), encoding="utf-8")

    with caplog.at_level("INFO"):
        exit_code = build_data_preparation_script.run()

    assert exit_code == 0
    assert "build-data-preparation" in caplog.text
