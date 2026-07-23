from __future__ import annotations

import pytest

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.amont_source import amont_series


def test_amont_series_returns_empty_dict_when_no_amont_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    rec = {"dossier": "melles", "stations_vigicrue_amont": []}

    assert amont_series(rec) == {}


def test_amont_series_uses_plain_name_for_single_amont(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    path = tmp_path / "touzac_g2_G2" / "amont_O770154002.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;3.100\n", encoding="utf-8-sig"
    )
    rec = {"dossier": "touzac_g2_G2", "stations_vigicrue_amont": ["O770154002"]}

    result = amont_series(rec)

    assert list(result.keys()) == ["debit_amont"]
    assert result["debit_amont"].tolist() == pytest.approx([3.100])


def test_amont_series_uses_code_suffix_for_multiple_amonts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    dossier_dir = tmp_path / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "amont_O001531001.csv").write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;1.000\n", encoding="utf-8-sig"
    )
    (dossier_dir / "amont_O005002001.csv").write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;2.000\n", encoding="utf-8-sig"
    )
    rec = {"dossier": "apas_G1_G4", "stations_vigicrue_amont": ["O001531001", "O005002001"]}

    result = amont_series(rec)

    assert set(result.keys()) == {"debit_amont_O001531001", "debit_amont_O005002001"}
    assert result["debit_amont_O001531001"].tolist() == pytest.approx([1.000])
    assert result["debit_amont_O005002001"].tolist() == pytest.approx([2.000])


def test_amont_series_returns_empty_series_when_csv_not_yet_local(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    rec = {"dossier": "touzac_g2_G2", "stations_vigicrue_amont": ["O770154002"]}

    result = amont_series(rec)

    assert result["debit_amont"].empty


def test_amont_series_deduplicates_repeated_code(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    path = tmp_path / "touzac_g2_G2" / "amont_O770154002.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;3.100\n", encoding="utf-8-sig"
    )
    rec = {
        "dossier": "touzac_g2_G2",
        "stations_vigicrue_amont": ["O770154002", "O770154002"],
    }

    result = amont_series(rec)

    assert list(result.keys()) == ["debit_amont"]


def test_amont_series_returns_empty_dict_when_key_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    rec = {"dossier": "melles"}

    assert amont_series(rec) == {}
