from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from projet_hydro.common import config
from projet_hydro.model.pipeline.predict_window import find_record, load_prediction_window


def test_find_record_returns_matching_dossier(tmp_path, monkeypatch):
    reference_dir = tmp_path / "REFERENCE"
    reference_dir.mkdir()
    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    records = [{"dossier": "a"}, {"dossier": "b", "facteur_debit": 0.95}]
    (reference_dir / "config-general.json").write_text(json.dumps(records), encoding="utf-8")

    result = find_record("b")

    assert result == {"dossier": "b", "facteur_debit": 0.95}


def test_find_record_raises_when_dossier_absent(tmp_path, monkeypatch):
    reference_dir = tmp_path / "REFERENCE"
    reference_dir.mkdir()
    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    (reference_dir / "config-general.json").write_text(json.dumps([{"dossier": "a"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="absent"):
        find_record("missing")


def test_load_prediction_window_covers_lookback_and_horizon(tmp_path, monkeypatch):
    reference_dir = tmp_path / "REFERENCE"
    centrales_dir = tmp_path / "centrales"
    nas_data_root = tmp_path / "nas_data"
    reference_dir.mkdir()
    (centrales_dir / "test_centrale").mkdir(parents=True)
    (nas_data_root / "test_centrale").mkdir(parents=True)

    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(config, "NAS_DATA_ROOT", nas_data_root)

    now = pd.Timestamp("2026-06-01 12:00:00")
    index = pd.date_range(now - pd.Timedelta(days=5), now, freq="1h", tz="UTC")
    debit_df = pd.DataFrame({"Date (TU)": index.strftime("%Y-%m-%dT%H:%M:%SZ"), "Valeur (en m³/s)": np.arange(len(index), dtype=float)})
    debit_path = centrales_dir / "test_centrale" / "O1234567890.csv"
    debit_df.to_csv(debit_path, sep=";", index=False)

    (centrales_dir / "test_centrale" / "bv.json").write_text(json.dumps({"stations_meteo_nwp": []}), encoding="utf-8")

    records = [{"dossier": "test_centrale", "flex_strategy": "DEFAULT", "station_vigicrue_reference": "O1234567890"}]
    (reference_dir / "config-general.json").write_text(json.dumps(records), encoding="utf-8")

    result = load_prediction_window("test_centrale", horizon_steps=8, timestep="hourly", now=now, lookback_days=5)

    assert not result.empty
    assert result.index.min() <= now - pd.Timedelta(days=4)


def test_load_prediction_window_frozen_reads_data_preparation_csv(tmp_path, monkeypatch):
    from projet_hydro.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv

    centrales_dir = tmp_path / "centrales"
    reference_dir = tmp_path / "REFERENCE"
    reference_dir.mkdir()
    dossier_dir = centrales_dir / "test_centrale"
    dossier_dir.mkdir(parents=True)

    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)

    records = [{"dossier": "test_centrale", "flex_strategy": "DEFAULT"}]
    (reference_dir / "config-general.json").write_text(json.dumps(records), encoding="utf-8")

    last_ts = pd.Timestamp("2026-05-01 10:00:00")
    index = pd.date_range(last_ts - pd.Timedelta(days=10), last_ts, freq="1h")
    df = pd.DataFrame({"debit_m3s": np.arange(len(index), dtype=float)}, index=index)
    write_data_preparation_csv(df, dossier_dir / "data_preparation.csv")

    result = load_prediction_window(
        "test_centrale", horizon_steps=8, timestep="hourly",
        now=pd.Timestamp("2026-07-23 09:00:00"),  # ignoré en mode frozen
        lookback_days=5, source="frozen",
    )

    assert not result.empty
    assert result.index.max() == last_ts  # ancré sur la dernière ligne connue, pas sur `now`
    assert result.index.min() >= last_ts - pd.Timedelta(days=5)


def test_load_prediction_window_frozen_returns_empty_when_no_data_preparation_csv(tmp_path, monkeypatch):
    centrales_dir = tmp_path / "centrales"
    reference_dir = tmp_path / "REFERENCE"
    reference_dir.mkdir()
    (centrales_dir / "test_centrale").mkdir(parents=True)

    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)
    (reference_dir / "config-general.json").write_text(
        json.dumps([{"dossier": "test_centrale"}]), encoding="utf-8"
    )

    result = load_prediction_window(
        "test_centrale", horizon_steps=8, timestep="hourly",
        now=pd.Timestamp("2026-07-23 09:00:00"), source="frozen",
    )

    assert result.empty
