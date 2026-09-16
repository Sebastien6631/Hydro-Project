from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "check_drift_script",
    Path(__file__).resolve().parents[2] / "cron" / "scripts" / "check-drift.py",
)
check_drift_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_drift_script)


def _setup(tmp_path, monkeypatch):
    from projet_hydro.common import config as cfg_mod
    from projet_hydro.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(check_drift_script, "REPORT_DIR", tmp_path / "logs" / "drift")


def _write_csv(tmp_path, dossier, debit_mean_shift_after_day=None, days=200):
    from projet_hydro.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv

    rng = np.random.default_rng(0)
    n = days * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    debit = rng.normal(10, 2, n)
    if debit_mean_shift_after_day is not None:
        cutoff = n - debit_mean_shift_after_day * 24
        debit[cutoff:] = rng.normal(40, 2, n - cutoff)  # dérive forte sur la fenêtre récente
    d = tmp_path / "centrales" / dossier
    d.mkdir(parents=True)
    write_data_preparation_csv(pd.DataFrame({"debit_m3s": debit}, index=idx), d / "data_preparation.csv")


def test_returns_0_and_writes_report_when_no_drift(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_csv(tmp_path, "touzac_g2_G2")

    assert check_drift_script.run() == 0
    report = json.loads((tmp_path / "logs" / "drift" / "touzac_g2_G2.json").read_text(encoding="utf-8"))
    assert report["dataset_drift"] is False
    assert (tmp_path / "logs" / "dvc_markers" / "check_drift.json").exists()


def test_detects_drift_but_still_returns_0(tmp_path, monkeypatch):
    # Une dérive détectée n'est PAS une erreur de pipeline (contrairement à validate-data.py).
    _setup(tmp_path, monkeypatch)
    _write_csv(tmp_path, "apas_G1_G4", debit_mean_shift_after_day=30)

    assert check_drift_script.run() == 0
    report = json.loads((tmp_path / "logs" / "drift" / "apas_G1_G4.json").read_text(encoding="utf-8"))
    assert report["dataset_drift"] is True


def test_short_history_is_skipped_not_an_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_csv(tmp_path, "touzac_g2_G2", days=10)  # pas assez pour référence + fenêtre récente

    assert check_drift_script.run() == 0
    assert not (tmp_path / "logs" / "drift" / "touzac_g2_G2.json").exists()


def test_no_data_is_not_an_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    (tmp_path / "centrales").mkdir()

    assert check_drift_script.run() == 0
