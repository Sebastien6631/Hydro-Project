from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "validate_data_script",
    Path(__file__).resolve().parents[2] / "cron" / "scripts" / "validate-data.py",
)
validate_data_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_data_script)


def _setup(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(validate_data_script, "REPORT_DIR", tmp_path / "logs" / "validation")


def _write_csv(tmp_path, dossier, debit):
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv

    idx = pd.date_range("2024-01-01", periods=len(debit), freq="1h")
    d = tmp_path / "centrales" / dossier
    d.mkdir(parents=True)
    write_data_preparation_csv(pd.DataFrame({"debit_m3s": debit}, index=idx), d / "data_preparation.csv")


def test_returns_0_and_writes_report_when_clean(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_csv(tmp_path, "apas_G1_G4", [10.0] * (200 * 24))

    assert validate_data_script.run() == 0
    report = json.loads((tmp_path / "logs" / "validation" / "apas_G1_G4.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert (tmp_path / "logs" / "dvc_markers" / "validate.json").exists()


def test_returns_1_on_contract_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_csv(tmp_path, "touzac_g2_G2", [-5.0] * (200 * 24))

    assert validate_data_script.run() == 1


def test_no_data_is_not_an_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    (tmp_path / "centrales").mkdir()

    assert validate_data_script.run() == 0
