from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "check_meteo_script",
    Path(__file__).resolve().parents[2] / "cron" / "scripts" / "check-meteo.py",
)
check_meteo_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_meteo_script)


def _setup(tmp_path, monkeypatch, dossiers=("apas_G1_G4", "touzac_g2_G2")):
    from projet_hydro.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    for d in dossiers:
        (tmp_path / "centrales" / d).mkdir(parents=True)
        (tmp_path / "centrales" / d / "bv.json").write_text(
            json.dumps({"stations_meteo_nwp": [{"id": 1, "lat": 44.5, "lon": 3.3}]}), encoding="utf-8")


def _frame():
    return pd.DataFrame({"temperature_S1": [280.0] * 9}, index=pd.date_range("2026-01-01", periods=9, freq="1h"))


def test_exit_0_when_every_centrale_has_a_forecast(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(check_meteo_script, "read_points", lambda points, start, end: _frame())

    assert check_meteo_script.run() == 0


def test_exit_1_and_names_the_centrale_when_open_meteo_returns_nothing(tmp_path, monkeypatch, caplog):
    """read_points ignore les points en échec et rend un DataFrame VIDE sans lever :
    c'est ce cas que le script doit transformer en échec explicite."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        check_meteo_script, "read_points",
        lambda points, start, end: pd.DataFrame(),
    )

    with caplog.at_level("ERROR"):
        code = check_meteo_script.run()

    assert code == 1
    assert "apas_G1_G4" in caplog.text and "touzac_g2_G2" in caplog.text


def test_dossier_without_meteo_station_is_skipped_not_failed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, dossiers=())
    (tmp_path / "centrales" / "sans_meteo").mkdir(parents=True)
    (tmp_path / "centrales" / "sans_meteo" / "bv.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(check_meteo_script, "read_points", lambda *a: (_ for _ in ()).throw(AssertionError("ne doit pas être appelé")))

    assert check_meteo_script.run() == 0
